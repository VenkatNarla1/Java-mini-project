/*
 * Copyright (C) 2019 Elias Yilma
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package cnn;

import UTIL.Mat;

public class MaxPool {

        public float[][][] input;  // [8] X [26] X [26]
    

        public float[][][] output;

    public float[][] max_pool(float[][] img) {
        float[][] pool = new float[img.length / 2][img[0].length / 2];
        for (int i = 0; i < pool.length - 1; i++) {
            for (int j = 0; j < pool[0].length - 1; j++) {
                pool[i][j] = Mat.m_max(Mat.m_sub(img, i * 2, i * 2 + 1, j * 2, j * 2 + 1));
            }
        }
        return pool;
    }

    public float[][][] forward(float[][][] dta) {
        input = dta;
        float[][][] result = new float[dta.length][dta[0].length][dta[0][0].length];
        for (int k = 0; k < dta.length; k++) {
            float[][] res = max_pool(dta[k]);
            result[k] = res;
        }
        output = result;
        return result;
    }

    public float[][][] backprop(float[][][] d_L_d_out) {
        float[][][] d_L_d_input = new float[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < output.length; i++) { // filter index 0 - 12 [13 values]
            for (int j = 0; j < output[0].length; j++) { //pool row index 0 -12 [13 values]
                for (int k = 0; k < output[0][0].length; k++) { //pool column index
                    //get 2X2 image region.
                    float[][] region = Mat.m_sub(input[i], j * 2, j * 2 + 1, k * 2, k * 2 + 1);
                    //loop through image region to get row,column index of the maximum value.
                    for (int m = 0; m < region.length; m++) {
                        for (int n = 0; n < region[0].length; n++) {
                            if (Math.abs(output[i][j][k] - region[m][n]) < 0.00000001) {
                                d_L_d_input[i][j * 2 + m][k * 2 + n] = d_L_d_out[i][j][k];
                            }
                        }
                    }
                }
            }
        }
        return d_L_d_input;
    }
}