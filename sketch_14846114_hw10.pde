// 14846114 曾敬貴 
float A = 0.35;          // 振幅
float f = 0.30;          // 頻率 (Hz)
float omega = TWO_PI * f;
float t = 0;             // 目前時間
float dt = 1.0 / 30.0;   // 30 FPS

// 場景座標
float wall_x = -1.2;
float eq_x = -0.2;       // 平衡位置
float y0 = 0.0;

float mass_w = 0.18;
float mass_h = 0.18;

void setup() {
  size(900, 350);
  frameRate(30);
}

void draw() {
  // 灰色背景
  background(#bfbfbf);
  
  // 座標轉換
  pushMatrix();
  translate(width/2, height/2);     // 原點移到中央
  scale(width / 3.0, -height / 3.0); // 縮放讓 x 範圍 ≈ [-1.5, 1.5]，y 軸向上
  translate(0, 0);
  
  // 白色的牆
  stroke(255);
  strokeWeight(0.02);
  line(wall_x, -0.35, wall_x, 0.35);
  
  // 目前位移
  float x_disp = A * cos(omega * t);
  float center_x = eq_x + x_disp;
  
  // 藍色質量塊（加深藍邊框）
  noStroke();
  fill(0, 0, 255);
  rectMode(CENTER);
  rect(center_x, y0, mass_w, mass_h);
  
  stroke(0, 0, 128);  // navy 邊框
  strokeWeight(0.006);
  noFill();
  rect(center_x, y0, mass_w, mass_h);
  
  // 彈簧：改用「鋸齒交替上下」方式
  float left_face = center_x - mass_w / 2;
  stroke(255);
  strokeWeight(0.009);  
  noFill();
  
  int coils = 30;     
  float amp = 0.07;   
  
  beginShape();
  for (int i = 0; i <= coils + 1; i++) {
    float bx = map(i, 0, coils + 1, wall_x, left_face);
    float by = y0;
    
    // 交替上下（奇偶判斷）
    if (i > 0 && i <= coils) {
      by += (i % 2 == 1 ? amp : -amp);
    }
    
    vertex(bx, by);
  }
  endShape();
  
  popMatrix();
  
  // 時間前進
  t += dt;
  
  // 10 秒後重頭
  if (t > 10) {
    t = 0;
  }
}
