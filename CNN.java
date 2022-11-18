package cnn;

import UTIL.Mat;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

public class CNN {

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }


    public static float[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        float[][] dta = new float[w][h];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            dta[row][col] = (((int) pixels[pixel] >> 16 & 0xff)) / 255.0f;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }
        }
        return dta;
    }

    public static float[][][] init_filters(int size) {
        float[][][] result = new float[size][3][3];
        for (int k = 0; k < size; k++) {
            result[k] = Mat.m_random(3, 3);
        }
        return result;
    }

    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        BufferedImage bi = load_image(final_path);
        return bi;
    }
    

    public static void train(int training_size) throws IOException {
        float[][][] filters = init_filters(8);
        int label_counter = 0;
        float ce_loss=0;
        int accuracy=0;
        float acc_sum=0.0f;
        float learn_rate=0.005f;
        
        //initialize layers
        Convolution conv=new Convolution();
        MaxPool pool=new MaxPool();
        SoftMax softmax=new SoftMax(13*13*8,10);

        float[][] out_l = new float[1][10];    
        for (int i = 0; i < training_size; i++) {
            //grab a random image from database.
            BufferedImage bi = mnist_load_random(label_counter);
            int correct_label = label_counter;
            if(label_counter==9){
                label_counter=0;
            }else{
                label_counter++;
            }
            
            float[][] pxl = img_to_mat(bi);
 
            float[][][] out = conv.forward(pxl, filters);


            out = pool.forward(out);
            
            out_l = softmax.forward(out); 
            
            // compute cross-entropy loss
            ce_loss += (float) -Math.log(out_l[0][correct_label]);
            accuracy += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            
            float[][] gradient=Mat.v_zeros(10);
            gradient[0][correct_label]=-1/out_l[0][correct_label];
            float[][][] sm_gradient=softmax.backprop(gradient,learn_rate);
            float[][][] mp_gradient=pool.backprop(sm_gradient);
            conv.backprop(mp_gradient, learn_rate);
            if(i % 100 == 99){
                System.out.println(" step: "+ i+ " loss: "+ce_loss/100.0+" accuracy: "+accuracy);
                ce_loss=0;
                acc_sum+=accuracy;
                accuracy=0;
            }
        }
        System.out.println("average accuracy:- "+acc_sum/training_size+"%");
    }

    public static void main(String[] args) throws IOException {      
        train(30000);
    }

}