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
const double DT_SEC = 0.02;

// Chirp signal parameters
const double CHIRP_START_FREQ = 0.1;  // Hz
const double CHIRP_END_FREQ = 5.0;    // Hz
const double CHIRP_DURATION = 30.0;   // seconds
const double CHIRP_AMPLITUDE = 0.5;   // cm/s
bool systemIdentificationMode = false;
unsigned long chirpStartTime = 0;

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
double K[] = {-12.1732, -11.3985, -59.1152, -13.8610};

// Potentiometer variables
int rawPotentiometerValue = 0;
double potentiometerValue = 0;

// Wait counter
unsigned int waitCounter = 0;

// Low-pass filter class
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

// Low-pass filters for velocities
LowPassFilter angularVelocityFilter(0.8647, 0.1353);
LowPassFilter cartVelocityFilter(0.8647, 0.1353);
LowPassFilter potentiometerFilter(0.1813, 0.8187);

// Position history for finite difference velocity calculation
double prevAngle1 = 0;
double prevAngle2 = 0;
double currAngle = 0;
double prevPos1 = 0;
double prevPos2 = 0;
double currPos = 0;

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
double generateChirpSignal();

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

    // Initialize previous values for finite differences
    currAngle = getAngleInRad();
    prevAngle1 = currAngle;
    prevAngle2 = currAngle;
    currPos = getPosInCm();
    prevPos1 = currPos;
    prevPos2 = currPos;

    delay(1000);

    xTaskCreatePinnedToCore(updateStepperSpeed, "updateStepperSpeed", 5000, NULL, 1, NULL, 0);
}

void loop() {
    if (micros() - currentTimeMicros > 20000) {
        currentTimeMicros = micros();
        switchIsPressed = !digitalRead(SWITCH_PIN);
        readPotentiometer();

        if (Serial.available()) {
            char c = Serial.read();
            if (c == 'i') {
                systemIdentificationMode = true;
                chirpStartTime = millis();
                Serial.println("Starting system identification chirp signal");
            }
        }

        switch (state) {
            case 0:
                stepper.setSpeed(cmToSteps(-10));
                if (switchIsPressed) {
                    stepper.setSpeed(0);
                    state = 1;
                    countEncPos = 0;
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
                if (abs(getAngleInRad()) < 0.01) {
                    state = 3;
                }
                break;

            case 3:
                double accInMetersPerSecondSquared = -(K[0] * getPosInCm() * 0.01 + K[1] * getCartVelocity() * 0.01 + K[2] * getAngleInRad() + K[3] * getAngularVelocity());
                double accInCmPerSecondSquared = accInMetersPerSecondSquared * 0.02 * 100;
                
                // Add chirp signal if in system identification mode
                if (systemIdentificationMode) {
                    accInCmPerSecondSquared += generateChirpSignal();
                }
                
                int newSpeed = stepper.speed() + cmToSteps(accInCmPerSecondSquared);
                stepper.setSpeed(newSpeed);

                if (abs(getAngleInRad()) > 0.5 || getPosInCm() < -17 || getPosInCm() > 17) {
                    stepper.setSpeed(0);
                    state = 0;
                    systemIdentificationMode = false;
                }
                break;
        }

        if (printFlag) {
            printAllStates();
        }
    }
}

double generateChirpSignal() {
    unsigned long currentTime = millis() - chirpStartTime;
    double t = currentTime / 1000.0;  // Convert to seconds
    
    if (t > CHIRP_DURATION) {
        systemIdentificationMode = false;
        return 0.0;
    }
    
    // Calculate instantaneous frequency (linear sweep)
    double freq = CHIRP_START_FREQ + (CHIRP_END_FREQ - CHIRP_START_FREQ) * (t / CHIRP_DURATION);
    
    // Generate chirp signal
    return CHIRP_AMPLITUDE * sin(2.0 * PI * (CHIRP_START_FREQ * t + 
                         0.5 * (CHIRP_END_FREQ - CHIRP_START_FREQ) * t * t / CHIRP_DURATION));
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
    return countEncTheta * RAD_PER_ENC + readPotentiometer() - 3.14;
}

double getPosInMm() {
    return MM_PER_ENC * countEncPos;
}

double getPosInCm() {
    return MM_PER_ENC * countEncPos / 10.0;
}

double getAngularVelocity() {
    // Update position history
    prevAngle2 = prevAngle1;
    prevAngle1 = currAngle;
    currAngle = getAngleInRad();
    
    // 2nd order central difference for velocity
    double rawSpeed = (currAngle - prevAngle2) / (2 * DT_SEC);
    
    return angularVelocityFilter.applyFilter(rawSpeed);
}

double getCartVelocity() {
    // Update position history
    prevPos2 = prevPos1;
    prevPos1 = currPos;
    currPos = getPosInCm();
    
    // 2nd order central difference for velocity
    double rawSpeed = (currPos - prevPos2) / (2 * DT_SEC);
    
    return cartVelocityFilter.applyFilter(rawSpeed);
}

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
    Serial.print(getAngularVelocity());
    Serial.print("\t");
    Serial.println(systemIdentificationMode ? generateChirpSignal() : 0.0);
}

void updateStepperSpeed(void *parameter) {
    disableCore0WDT();
    while (true) {
        stepper.runSpeed();
    }
}