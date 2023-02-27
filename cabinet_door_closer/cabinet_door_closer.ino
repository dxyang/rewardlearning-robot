#define motorPin 9
#define switchPin 4
// #define pullVal 110
#define pushVal 110
#define middleVal 192
// #define pushVal 244
#define pullVal 244

int lastSwitchVal; 

void setup() {
  // put your setup code here, to run once:
  pinMode(motorPin, OUTPUT);
  pinMode(switchPin, INPUT_PULLUP);
  Serial.begin(9600);

  lastSwitchVal = digitalRead(switchPin);
}

void test_switch() {
  int val = digitalRead(switchPin);
  Serial.println(val);
  // switch = 1 means unpressed, switch = 0 means pressed
}

void loop() {
  /*
  0 = waiting mode
  1 = switch was already hit
  2 = timeout
  3 = button hit or timeout
  4 = done
  */
  Serial.println("0");
  //clear input buffer
  while (Serial.available() > 0) {
    Serial.read();
  }

  // wait for user input (time to unspool for)
  while (Serial.available() == 0){}  
  
  // get user input
  long waitTime = Serial.parseInt();
  // Serial.println(waitTime);
  
  // if the button is already pressed, don't do anything
  // as the door is already closed
  int val = digitalRead(switchPin);
  if (val == 0) {
    Serial.println("1");
    return;
  } 

  // keep motor pulling in until timeout or button press read
  long timeout = 20000;
  unsigned long startTime = millis();
  while(digitalRead(switchPin) == 1){
    if (millis() - startTime > timeout) {
      Serial.println("2");
      break;
    }
    analogWrite(motorPin, pullVal);
  }
  Serial.println("3");

  // unspool motor for waitTime 
  startTime = millis();
  while(millis() - startTime < waitTime){
    analogWrite(motorPin,pushVal);
  }
  analogWrite(motorPin,middleVal);
  Serial.println("4");
}