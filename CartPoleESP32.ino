#include "AccelStepper.h"

#define DIR_PIN 25
#define PUL_PIN 26

#define ENC_X_A_PIN 27
#define ENC_X_B_PIN 14

#define ENC_TH_A_PIN 16
#define ENC_TH_B_PIN 17 // 23

#define SWITCH_PIN 0 // 0

double pullyDiameterInMm = 37.8;
double lengthOfPullyInMm = (pullyDiameterInMm/2)*PI*2;
double normalStepsPerRev = 200;
double p = 16;
double microsteppingRes = 200 * p;
double MM_PER_STEP = lengthOfPullyInMm/microsteppingRes;
double MM_PER_ENC = lengthOfPullyInMm/600;
double RAD_PER_ENC = 0.003067962;
double dtInSec = 0.02;

int countEncTheta = 0;
int countEncPos = 0;
int buttonState = 0;
AccelStepper stepper = AccelStepper(1, PUL_PIN, DIR_PIN);


void IRAM_ATTR isrEncPos() {
  if(digitalRead(ENC_X_B_PIN)){
    countEncPos++;
  }else{
    countEncPos--;
  }
}

void IRAM_ATTR isrEncTheta() {
  if(digitalRead(ENC_TH_B_PIN)){
    countEncTheta++;
  }else{
    countEncTheta--;
  }
}

int CmToSteps(double l_cm){
  double steps = (l_cm * 10)/MM_PER_STEP;
  int rounded_steps = steps;
  return -rounded_steps;
}

double StepsToCm(int steps){
  return MM_PER_STEP*steps;
}

double StepsToMeters(int steps){
  return StepsToCm(steps)*0.01;
}

double getAngleInRad(){
  return countEncTheta*RAD_PER_ENC;
}
double getPosInMm(){
  return MM_PER_ENC*countEncPos;
}
double getPosInCm(){
  return MM_PER_ENC*countEncPos/10.0;
}
// =========== ANGULAR VELOCITY ======================
double angle_yk_0 = 0;
double angle_yk_1 = 0;
double angle_uk_0 = 0;
double angle_uk_1 = 0;
double prevAngle = 0;
double currAngle = 0;
double AngularVelocityRunThroughLowPassFilter(double angle_uk_0){
 
  angle_yk_0 = 0.8647*angle_uk_1 + angle_yk_1*0.1353;
  angle_yk_1 = angle_yk_0;
  angle_uk_1 = angle_uk_0;

  return angle_yk_0; 
}

double getAngularVelocity(){
  currAngle = getAngleInRad();
  double rawSpeed = (currAngle - prevAngle)/dtInSec;
  prevAngle = currAngle;
  return AngularVelocityRunThroughLowPassFilter(rawSpeed);
}

//===================================================
// =========== CART VELOCITY ======================
double pos_yk_0 = 0;
double pos_yk_1 = 0;
double pos_uk_0 = 0;
double pos_uk_1 = 0;
double prevPos = 0;
double currPos = 0;
double CartVelocityRunThroughLowPassFilter(double pos_uk_0){
 
  pos_yk_0 = 0.8647*pos_uk_1 + pos_yk_1*0.1353;
  pos_yk_1 = pos_yk_0;
  pos_uk_1 = pos_uk_0;

  return pos_yk_0; 
}

double getCartVelocity(){
  currPos = getPosInCm();
  double rawSpeed = (currPos - prevPos)/dtInSec;
  prevPos = currPos;
  return CartVelocityRunThroughLowPassFilter(rawSpeed);
}

//===================================================

TaskHandle_t Task1;
unsigned int  now = 0;

void setup(){  
  Serial.begin(115200);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(PUL_PIN, OUTPUT);

  pinMode(SWITCH_PIN, INPUT_PULLUP);

  pinMode(ENC_TH_A_PIN,INPUT);
  pinMode(ENC_TH_B_PIN,INPUT);
  attachInterrupt(ENC_TH_A_PIN, isrEncTheta, RISING);

  pinMode(ENC_X_A_PIN,INPUT);
  pinMode(ENC_X_B_PIN,INPUT);
  attachInterrupt(ENC_X_A_PIN, isrEncPos, RISING);

  stepper.setMaxSpeed(CmToSteps(30));
  stepper.setAcceleration(CmToSteps(15));     
  countEncTheta = 0;
  countEncPos = 0;
  xTaskCreatePinnedToCore(    codeForTask1,    "StepperRun",    5000,      NULL,    1,    &Task1,    0);
  now = millis();

}
int stepMode = 1;
int pos = CmToSteps(10);

int first = 0;
int second = 0;
int print = 1;
int state = 0;
int dt_micros = 20000;
unsigned int currentTimeMicros = 0;
bool switchIsPressed = false;

void loop(){
  if(micros() - currentTimeMicros > dt_micros){
    currentTimeMicros = micros();
    switchIsPressed = !digitalRead(SWITCH_PIN);
   
    if(state == 0){
      stepper.setSpeed(CmToSteps(-10));
      if(switchIsPressed){
        stepper.setSpeed(0);
        state = 1;
        countEncPos = 0;
      }
    }

    else if(state == 1){
      if(getPosInCm() < 20){
        stepper.setSpeed(CmToSteps(30));
      }else{
        stepper.setSpeed(0);
      }
    }

    if(print){
      // Serial.print(getAngularVelocity());
      // Serial.print("\t");
      printAllStates();
    }

  }
}

void printAllStates(){
    Serial.print(getPosInCm());
    Serial.print("\t");
    Serial.print(getCartVelocity());
    Serial.print("\t");
    Serial.print(getAngleInRad());
    Serial.print("\t");
    Serial.println(getAngularVelocity());

}
void codeForTask1( void * parameter ) {
  disableCore0WDT();
   while(true){
      // second++;
      stepper.runSpeed();

   }
}