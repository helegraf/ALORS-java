package alors.latent_features;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tests {@link WEKAFeaturePredictor}.
 * 
 * @author helegraf
 *
 */
public class WEKAFeaturePredictorTest {
	
	private Logger logger = LoggerFactory.getLogger(WEKAFeaturePredictorTest.class);

	/**
	 * Tests {@link WEKAFeaturePredictor#train(double[][], double[][])} on a small
	 * example.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testTrain() throws Exception {
		WEKAFeaturePredictor predictor = new WEKAFeaturePredictor();
		double[][] featureMatrixX = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
		double[][] featureMatrixU = { { 1, 2 }, { 4, 5 }, { 7, 8 } };
		predictor.train(featureMatrixX, featureMatrixU);
		
		assertEquals(true, predictor.isPrepared());
	}

	/**
	 * Tests {@link WEKAFeaturePredictor#predict(double[])} on a small example.
	 * 
	 * @throws Exception
	 */
	@Test
	public void testTrainAndPredict() throws Exception {
		WEKAFeaturePredictor predictor = new WEKAFeaturePredictor();
		double[][] featureMatrixX = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
		double[][] featureMatrixU = { { 1, 2 }, { 4, 5 }, { 7, 8 } };

		predictor.train(featureMatrixX, featureMatrixU);
		double[] prediction = predictor.predict(new double[] { 4, 5, 6 });

		logger.info("Expected result: [4.0, 5.0]");
		logger.info("Actual result: {}", Arrays.toString(prediction));
		
		// basic output assertions
		assertArrayEquals(new double[] {4, 5}, prediction, 2.0);
	}
}
