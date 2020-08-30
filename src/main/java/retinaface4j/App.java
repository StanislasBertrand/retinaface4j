package retinaface4j;

import java.util.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.image.WritableRaster;
import java.util.Map;
import java.util.HashMap;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.BasicStroke;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import retinaface4j.ArrayUtils;
import retinaface4j.PriorBox;
import retinaface4j.Utils;
import retinaface4j.Detector;

public class App {
    public static void main(String[] args) {

        // Load arguments
        String modelPath = null;
        String imgPath = null;
        String outputPath = null;
        Double detThresh = 0.9;
        Double nmsThresh = 0.4;
        try {
            modelPath = args[0];
            imgPath = args[1];
            outputPath = args[2];
        } catch (Exception e) {
            System.out.println("[ERROR] could not read arguments");
            System.out.println(e.getMessage());
        }

        // Load model
        Detector detector = new Detector(modelPath, detThresh, nmsThresh);

        // Read image
        BufferedImage img = null;
        INDArray nd4jimg = null;
        try {
            img = ImageIO.read(new File(imgPath));
            NativeImageLoader loader = new NativeImageLoader(img.getHeight(), img.getWidth(), 3);
            nd4jimg = loader.asMatrix(img);
        } catch (IOException e) {
            System.out.println("[ERROR] could not read image");
            System.out.println(e.getMessage());
        }

        // predict
        INDArray dets = detector.predict(nd4jimg);

        // generate output image
        Graphics2D g2d = img.createGraphics();
        g2d.setColor(Color.RED);
        for (int i = 0; i<dets.shape()[0]; i++) {
            g2d.drawRect(dets.getInt(i,0), dets.getInt(i,1), dets.getInt(i,2)-dets.getInt(i,0), dets.getInt(i,3)-dets.getInt(i,1));
        }
        g2d.dispose();
        try {
            ImageIO.write(img, "png", new File(outputPath));
        } catch (Exception e) {
            System.out.println("[ERROR] Could not save image.");
            System.out.println(e.getMessage());
        }

        System.exit(0);
   }
}