package retinaface4j;

import java.util.*;
import java.lang.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class PriorBox{

    public static List<Integer> steps = Arrays.asList(8,16,32);
    public static List<Integer> minSizes = Arrays.asList(16,32,64,128,256,512);

    private int imgWidth;
    private int imgHeight;
    private int[][] ftMaps;

    public PriorBox(int w, int h){
        this.setW(w);
        this.setH(h);
        int[][] ftMaps = new int[3][2];
        for (int i=0; i<3; i++){
            ftMaps[i][0] = (int) Math.ceil((float)h/steps.get(i));
            ftMaps[i][1] = (int) Math.ceil((float)w/steps.get(i));
        }
        this.setFtMaps(ftMaps);
    }

    public void setW(int w) {
        this.imgWidth = w;
    }
    public void setH(int h){
        this.imgHeight = h;
    }
    public void setFtMaps(int[][] maps){
        this.ftMaps = maps;
    }

    public Tensor forward() {
        List<Float> anchors = new ArrayList<Float>();
        for (int k=0; k<this.ftMaps.length; k++){
            float ms = (float) minSizes.get(2*k);
            int ftMapSz1 = ftMaps[k][0];
            int ftMapSz2 = ftMaps[k][1];
            for (int i=0; i<ftMapSz1; i++){
                for (int j=0; j<ftMapSz2; j++){
                    float s_kx = ms / this.imgWidth;
                    float s_ky = ms / this.imgHeight;
                    float dense_cx = (j+0.5f)*steps.get(k)/this.imgWidth;
                    float dense_cy = (i+0.5f)*steps.get(k)/this.imgHeight;
                    anchors.add(dense_cx);
                    anchors.add(dense_cy);
                    anchors.add(s_kx);
                    anchors.add(s_ky);
                    float s_kx2 = 2*ms / this.imgWidth;
                    float s_ky2 = 2*ms / this.imgHeight;
                    anchors.add(dense_cx);
                    anchors.add(dense_cy);
                    anchors.add(s_kx2);
                    anchors.add(s_ky2);
                }
            }
        }

        float[] anchorsArr = new float[anchors.size()];
        int index = 0;
        for (final Float value: anchors) {
           anchorsArr[index++] = value;
        }
        Tensor output =
            Tensor.fromBlob(
                anchorsArr,
                new long[] {anchors.size()/4, 4});

        return output;
    }
}