package alors.latent_features;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

public class WEKAFeaturePredictorTest {

	@Test
	public void testTrain() throws Exception {
		WEKAFeaturePredictor predictor = new WEKAFeaturePredictor();
		double[][] featureMatrixX = {{1,2,3},{4,5,6},{7,8,9}};
		double[][] featureMatrixU = {{1,2},{3,4},{5,6}};
		predictor.train(featureMatrixX, featureMatrixU);
	}
	
	@Test
	public void testTrainAndPredict() throws Exception {
		WEKAFeaturePredictor predictor = new WEKAFeaturePredictor();
		double[][] featureMatrixX = {{1,2,3},{4,5,6},{7,8,9}};
		double[][] featureMatrixU = {{1,2},{4,5},{7,8}};
		
		predictor.train(featureMatrixX, featureMatrixU);		
		double[] prediction = predictor.predict(new double[] {4,5,6});
		
		System.out.println("Predicted: " + Arrays.toString(prediction));
		System.out.println("Expected: [4, 5]");
	}
}
