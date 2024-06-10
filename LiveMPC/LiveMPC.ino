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
// double K[] = {-0.8825, -1.8860, -25.2267, -4.9230};
//double K[] = {-2.7602, -3.5188, -30.4287, -5.8626};
//double K[] = {-8.5424, -7.5762, -42.3058, -8.0055};
//double K[] =  {-8.7374, -8.4591, -49.5014, -11.5584}; // Q = diag([100,1,100,1]);l_tot = 0.47; m = 0.135;
//double K[] = {  -12.1732 ,-11.3985,  -59.1152,  -13.8610};
double K[] = {-6.2367, -6.4931, -42.3132, -10.0260};
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

// Previous positions for velocity calculation
double prevAngle = 0;
double currAngle = 0;
double prevPos = 0;
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
    Serial.begin(921600);
    
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
int test = 0;
unsigned int now = 0;
void loop() {
    if (micros() - currentTimeMicros > 20000) {
        printAllStates();

        currentTimeMicros = micros();
        switchIsPressed = !digitalRead(SWITCH_PIN);
        readPotentiometer();

        while(Serial.available() <= 0){}
        String input = Serial.readStringUntil('\n');
        double commandValue = input.toDouble(); 

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
                double accInMetersPerSecondSquared = commandValue;
                double accInCmPerSecondSquared = accInMetersPerSecondSquared * 0.02 * 100;
                int newSpeed = stepper.speed() + cmToSteps(accInCmPerSecondSquared);
                stepper.setSpeed(newSpeed);

                if (abs(getAngleInRad()) > 0.5 || getPosInCm() < -17 || getPosInCm() > 17) {
                    stepper.setSpeed(0);
                    state = 0;
                }
                break;

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
    return countEncTheta * RAD_PER_ENC + readPotentiometer() - 3.14;
}

double getPosInMm() {
    return MM_PER_ENC * countEncPos;
}

double getPosInCm() {
    return MM_PER_ENC * countEncPos / 10.0;
}

double getAngularVelocity() {
    currAngle = getAngleInRad();
    double rawSpeed = (currAngle - prevAngle) / DT_SEC;
    prevAngle = currAngle;
    return angularVelocityFilter.applyFilter(rawSpeed);
}

double getCartVelocity() {
    currPos = getPosInCm();
    double rawSpeed = (currPos - prevPos) / DT_SEC;
    prevPos = currPos;
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
    Serial.println(getAngularVelocity());
}

void updateStepperSpeed(void *parameter) {
    disableCore0WDT();
    while (true) {
        stepper.runSpeed();
    }
}
