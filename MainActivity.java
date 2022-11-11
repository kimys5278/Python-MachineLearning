package com.cookandroid.mobileprogramminghomework;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.ColorStateList;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TabHost;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {
    Button btn1,btn2,btn3;
    RadioGroup rg1;
    RadioButton rgb1,rgb2,rgb3;
    TextView text1,text2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle("탭호스트 과제");

        btn1 = (Button)findViewById(R.id.btn1);
        btn2 = (Button)findViewById(R.id.btn2);
        btn3 = (Button)findViewById(R.id.btn3);
        rg1 = (RadioGroup) findViewById(R.id.rg1);
        rgb1 = (RadioButton) findViewById(R.id.rgb1);
        rgb2 = (RadioButton) findViewById(R.id.rgb2);
        rgb3 = (RadioButton) findViewById(R.id.rgb3);
        text1 = (TextView) findViewById(R.id.text1);
        text2 = (TextView) findViewById(R.id.text2);

        TabHost tabHost = findViewById(R.id.tabHost);
        tabHost.setup();

        TabHost.TabSpec tabSpecColor = tabHost.newTabSpec("Color").setIndicator("색상");
        tabSpecColor.setContent(R.id.색상);
        tabHost.addTab(tabSpecColor);

        TabHost.TabSpec tabSpecSize = tabHost.newTabSpec("Size").setIndicator("크기");
        tabSpecSize.setContent(R.id.크기);
        tabHost.addTab(tabSpecSize);

        tabHost.setCurrentTab(0);

        rg1.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup radioGroup, int i) {
                switch (rg1.getCheckedRadioButtonId()) {
                    case R.id.rgb1:
                        text1.setTextColor(ColorStateList.valueOf(Color.parseColor("#F41616")));
                        break;
                    case R.id.rgb2:
                        text1.setTextColor(ColorStateList.valueOf(Color.parseColor("#1BCF22")));
                        break;
                    case R.id.rgb3:
                        text1.setTextColor(ColorStateList.valueOf(Color.parseColor("#0A30EC")));
                        break;
                }
                btn1.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        text2.setTextSize(40);
                    }
                });
                btn2.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        text2.setTextSize(60);
                    }
                });
                btn3.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        text2.setTextSize(80);
                    }
                });
            }

        });


    }
}




