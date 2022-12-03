package com.example.neonintrinsics;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.*;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        System.loadLibrary( "opencv_java4");
    }

    public static double convpix(double[][] input,
                                                int x, int y,
                                                double[][] k,
                                                int kernelWidth,
                                                int kernelHeight) {
        double output = 0;
        for (int i = 0; i < kernelWidth; ++i) {
            for (int j = 0; j < kernelHeight; ++j) {
                output = output + (input[x + i][y + j] * k[i][j]);
            }
        }
        return output;
    }

    public static double[][] conv2d(double[][] input,
                                           int width, int height,
                                           double[][] kernel,
                                           int kernelWidth,
                                           int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        double[][] output = new double[smallWidth][smallHeight];
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = 0;
            }
        }
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = convpix(input, i, j, kernel,
                        kernelWidth, kernelHeight);
            }
        }
        return output;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());


       Mat dst = new Mat(128,128,CvType.CV_32F);
        Mat img = new Mat(128,128,CvType.CV_32F);
      /* load lena into img */
       float ker[] = new float[]{ 1.0f/16.0f,2.0f/16.0f,1.0f/16.0f,2.0f/16.0f,4.0f/16.0f,2.0f/16.0f,1.0f/16.0f,2.0f/16.0f,1.0f/16.0f};
       double kerd[][] = new double[3][3];
       for(int i=0;i<0;i++){
           kerd[i/3][i%3] = (double)ker[i];
       }
      Mat kernel = new Mat(3,3, CvType.CV_32F);
       kernel.put(0,0,ker);
       long startTime = System.nanoTime();
      Imgproc.filter2D(img, dst, -1, kernel);
    long endTime = System.nanoTime();
       long timeElapsed = endTime - startTime;
      TextView tv_ocv = findViewById(R.id.text_ocv);
      String txt = "opencv time: " + timeElapsed/1000 +"us\n";
      double[][] imgd = new double[130][130]; // assume padded
        startTime = System.nanoTime();
      conv2d(imgd,130,130,kerd,3,3);
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        txt += "java time: " + timeElapsed/1000 +"us\n";
     tv_ocv.setText(txt);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
