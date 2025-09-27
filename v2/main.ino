#include "AccelStepper.h"

// Pin definitions
#define DIR_PIN 25
#define PUL_PIN 26
#define ENC_X_A_PIN 27
#define ENC_X_B_PIN 14
#define ENC_TH_A_PIN 16
#define ENC_TH_B_PIN 17
#define SWITCH_PIN 0
#define POTMET_PIN 34

// Constants
const double PULLEY_DIAMETER_MM = 37.8;
const double PULLEY_LENGTH_MM = (PULLEY_DIAMETER_MM / 2) * PI * 2;
const double NORMAL_STEPS_PER_REV = 200;
const double MICROSTEPPING_RES = NORMAL_STEPS_PER_REV * 16;
const double MM_PER_STEP = PULLEY_LENGTH_MM / MICROSTEPPING_RES;
const double MM_PER_ENC = PULLEY_LENGTH_MM / 600;
const double RAD_PER_ENC = 0.00306796157;
const double DT_SEC = 0.02;  // fallback DT

// Encoder counts
volatile int countEncTheta = 0;
volatile int countEncPos = 0;

// Stepper motor
AccelStepper stepper(AccelStepper::DRIVER, PUL_PIN, DIR_PIN);

// State variables
int printFlag = 1;
int state = 0;
unsigned int currentTimeMicros = 0;
bool switchIsPressed = false;

// LQR gains
double K[] = {  -12.1732 ,-11.3985,  -59.1152,  -13.8610};

// Potentiometer variables
int rawPotentiometerValue = 0;
double potentiometerValue = 0;

// Wait counter
unsigned int waitCounter = 0;

// ---------------- Filters / Derivatives ----------------
class LowPassFilter {
public:
    LowPassFilter(double alpha, double beta) : alpha(alpha), beta(beta), yk_0(0), yk_1(0), uk_0(0), uk_1(0) {}
    double applyFilter(double uk_0) {
        yk_0 = alpha * uk_1 + beta * yk_1;
        yk_1 = yk_0;
        uk_1 = uk_0;
        return yk_0;
    }
private:
    double alpha;
    double beta;
    double yk_0;
    double yk_1;
    double uk_0;
    double uk_1;
};

// Tustin (dirty) differentiator
class DirtyDerivativeTustin {
public:
    explicit DirtyDerivativeTustin(double tau_s) : tau(tau_s) {}
    void setTau(double tau_s){ tau = tau_s; }
    void reset(double x0 = 0.0){ initialized = false; v = 0.0; x_prev = x0; }
    double update(double x_now, double dt_s){
        if (!initialized){
            initialized = true; x_prev = x_now; v = 0.0; return 0.0;
        }
        double denom = 2.0 * tau + dt_s;
        if (denom <= 1e-9) return v;
        double a = (2.0 * tau - dt_s) / denom;
        double b = 2.0 / denom;
        v = a * v + b * (x_now - x_prev);
        x_prev = x_now;
        return v;
    }
private:
    double tau;         // [s]
    double v = 0.0;     // derivative output
    double x_prev = 0.0;
    bool initialized = false;
};

// Only keep LPF for potentiometer if you still need it
LowPassFilter potentiometerFilter(0.1813, 0.8187);

// Previous positions for velocity calculation (no longer used directly)
double prevAngle = 0;
double currAngle = 0;
double prevPos = 0;
double currPos = 0;

// --- Tustin differentiators (tune taus) ---
DirtyDerivativeTustin dtheta(0.03); // rad/s, tau ~ 30–50 ms
DirtyDerivativeTustin dx(0.08);     // cm/s,  tau ~ 70–120 ms

// --- Timing for variable dt
double dt_s = DT_SEC;
unsigned long prevTickMicros = 0;
inline double updateDt(){
    unsigned long now = micros();
    if (prevTickMicros == 0) { prevTickMicros = now; return dt_s; }
    dt_s = (now - prevTickMicros) * 1e-6;
    prevTickMicros = now;
    if (dt_s <= 0.0 || dt_s > 0.1) dt_s = DT_SEC; // guard
    return dt_s;
}

// Function prototypes
int cmToSteps(double l_cm);
double stepsToCm(int steps);
double stepsToMeters(int steps);
double getAngleInRad();
double getPosInMm();
double getPosInCm();
double getAngularVelocity();
double getCartVelocity();
void waitAndGoToNextStep(int nextState);
float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh);
double readPotentiometer();
void printAllStates();
void updateStepperSpeed(void *parameter);

// Interrupt service routines
void IRAM_ATTR isrEncPos() {
    if (digitalRead(ENC_X_B_PIN)) {
        countEncPos++;
    } else {
        countEncPos--;
    }
}

void IRAM_ATTR isrEncTheta() {
    if (digitalRead(ENC_TH_B_PIN)) {
        countEncTheta++;
    } else {
        countEncTheta--;
    }
}

void setup() {
    Serial.begin(115200);

    pinMode(DIR_PIN, OUTPUT);
    pinMode(PUL_PIN, OUTPUT);

    pinMode(POTMET_PIN, INPUT);
    pinMode(SWITCH_PIN, INPUT_PULLUP);

    pinMode(ENC_TH_A_PIN, INPUT);
    pinMode(ENC_TH_B_PIN, INPUT);
    attachInterrupt(digitalPinToInterrupt(ENC_TH_A_PIN), isrEncTheta, RISING);

    pinMode(ENC_X_A_PIN, INPUT);
    pinMode(ENC_X_B_PIN, INPUT);
    attachInterrupt(digitalPinToInterrupt(ENC_X_A_PIN), isrEncPos, RISING);

    stepper.setMaxSpeed(cmToSteps(70));
    stepper.setAcceleration(cmToSteps(20));

    countEncTheta = 0;
    countEncPos = 0;

    delay(1000);

    xTaskCreatePinnedToCore(updateStepperSpeed, "updateStepperSpeed", 5000, NULL, 1, NULL, 0);
}

void loop() {
    if (micros() - currentTimeMicros > 20000) {
        currentTimeMicros = micros();
        updateDt(); // measure actual Ts for differentiators

        switchIsPressed = !digitalRead(SWITCH_PIN);
        readPotentiometer();

        switch (state) {
            case 0:
                stepper.setSpeed(cmToSteps(-10));
                if (switchIsPressed) {
                    stepper.setSpeed(0);
                    state = 1;
                    countEncPos = 0;
                    // reset differentiators to avoid spikes
                    dtheta.reset(getAngleInRad());
                    dx.reset(getPosInCm());
                }
                break;

            case 1:
                if (getPosInCm() < 20) {
                    stepper.setSpeed(cmToSteps(30));
                } else {
                    stepper.setSpeed(0);
                    waitAndGoToNextStep(2);
                }
                break;

            case 2:
                countEncPos = 0;
                // reset differentiators when re-zeroing
                dtheta.reset(getAngleInRad());
                dx.reset(getPosInCm());
                if (abs(getAngleInRad()) < 0.01) {
                    state = 3;
                }
                break;

            case 3: {
                double accInMetersPerSecondSquared = -(K[0] * getPosInCm() * 0.01
                                                     + K[1] * getCartVelocity() * 0.01
                                                     + K[2] * getAngleInRad()
                                                     + K[3] * getAngularVelocity());
                double accInCmPerSecondSquared = accInMetersPerSecondSquared * 0.02 * 100;
                int newSpeed = stepper.speed() + cmToSteps(accInCmPerSecondSquared);
                stepper.setSpeed(newSpeed);

                if (abs(getAngleInRad()) > 0.5 || getPosInCm() < -17 || getPosInCm() > 17) {
                    stepper.setSpeed(0);
                    state = 0;
                }
            } break;
        }

        if (printFlag) {
            printAllStates();
        }
    }
}

int cmToSteps(double l_cm) {
    double steps = (l_cm * 10) / MM_PER_STEP;
    return -static_cast<int>(steps);
}

double stepsToCm(int steps) {
    return MM_PER_STEP * steps;
}

double stepsToMeters(int steps) {
    return stepsToCm(steps) * 0.01;
}

double getAngleInRad() {
    // Keep your existing biasing with potentiometer (optional)
    return countEncTheta * RAD_PER_ENC + readPotentiometer() - 3.14;
}

double getPosInMm() {
    return MM_PER_ENC * countEncPos;
}

double getPosInCm() {
    return MM_PER_ENC * countEncPos / 10.0;
}

// ----- NEW: Tustin-based velocities -----
double getAngularVelocity() {
    // Output in rad/s
    return dtheta.update(getAngleInRad(), dt_s);
}

double getCartVelocity() {
    // Output in cm/s
    return dx.update(getPosInCm(), dt_s);
}
// ----------------------------------------

void waitAndGoToNextStep(int nextState) {
    waitCounter++;
    if (waitCounter == 25) {
        state = nextState;
        waitCounter = 0;
    }
}

float mapFloat(float value, float fromLow, float fromHigh, float toLow, float toHigh) {
    return (value - fromLow) * (toHigh - toLow) / (fromHigh - fromLow) + toLow;
}

double readPotentiometer() {
    rawPotentiometerValue = analogRead(POTMET_PIN);
    potentiometerValue = mapFloat(rawPotentiometerValue, 0, 4096, -0.1, 0.1);
    return potentiometerFilter.applyFilter(potentiometerValue);
}

void printAllStates() {
    Serial.print(getPosInCm());
    Serial.print("\t");
    Serial.print(getCartVelocity());
    Serial.print("\t");
    Serial.print(getAngleInRad());
    Serial.print("\t");
    Serial.println(getAngularVelocity());
}

void updateStepperSpeed(void *parameter) {
    disableCore0WDT();
    while (true) {
        stepper.runSpeed();
    }
}
