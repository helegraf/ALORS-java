package alors;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import alors.latent_features.FeaturePredictorException;
import alors.matrix_completion.MatrixCompleterException;
import alors.matrix_completion.cofirank.CofiConfig;
import alors.matrix_completion.cofirank.CofirankCPlusPlus;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * Testing {@link Alors}.
 * 
 * @author helegraf
 *
 */
public class AlorsTest {

	private Logger logger = LoggerFactory.getLogger(AlorsTest.class);

	@AfterEach
	public void cleanUp() throws IOException {
		FileUtils.cleanDirectory(Paths.get("cofirank", "config").toFile());
		FileUtils.deleteQuietly(Paths.get("cofirank", "data", "train.lsvm").toFile());
		FileUtils.deleteQuietly(Paths.get("cofirank", "data", "test.lsvm").toFile());
		FileUtils.cleanDirectory(Paths.get("cofirank", "default_out").toFile());
	}

	/**
	 * Tests the basic functionality of {@link Alors} by solving a small dataset.
	 * Uses {@link CofirankCPlusPlus}, so only available on the OS cofirank is
	 * compiled for, default linux.
	 * 
	 * @throws IOException
	 * @throws MatrixCompleterException
	 * @throws FeaturePredictorException
	 * @throws AlorsException 
	 */
	@Test
	@EnabledOnOs(OS.LINUX)
	public void testALORSExecution() throws IOException, MatrixCompleterException, FeaturePredictorException, AlorsException {
		// read some instances
		BufferedReader reader = new BufferedReader(
				new FileReader(Paths.get("src", "test", "resources", "noProbing_nonan_noid.arff").toString()));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		Instances train = new Instances(data, 0, data.numInstances() - 10);
		Instances test = new Instances(data, data.numInstances() - 10, 10);

		// split into X (instance features) and M (instance w\ algorithm performances)
		double[][] x_train = getPortion(train, 22, false);
		double[][] m_train = getPortion(train, 22, true);
		double[][] x_test = getPortion(test, 22, false);
		double[][] m_test = getPortion(test, 22, true);

		// train
		String executablePath = Paths.get("cofirank", "dist", "cofirank-deploy").toString();
		String configurationPath = Paths.get("cofirank", "config", "default.cfg").toString();
		String outFolderPath = Paths.get("cofirank", "default_out").toString();
		String trainFilePath = Paths.get("cofirank", "data", "train.lsvm").toString();
		String testFilePath = Paths.get("cofirank", "data", "test.lsvm").toString();

		CofiConfig config = new CofiConfig(executablePath, configurationPath, outFolderPath, trainFilePath,
				testFilePath);
		CofirankCPlusPlus cofirank = new CofirankCPlusPlus(config);
		Alors alors = new Alors(cofirank);

		alors.completeMatrixAndPrepareColdStart(m_train, x_train);
		
		assertEquals(true, alors.isPrepared());

		// test
		double avgRmse = 0;
		for (int i = 0; i < x_test.length; i++) {
			double[] prediction = alors.predictForFeatures(x_test[i]);
			logger.debug("Expected values: {}", m_test[i]);
			logger.debug("Actual values: {}", prediction);
			
			// assert roughly sensible predictions
			double rmse = 0;
			for (int j = 0;j < prediction.length; j++) {
				rmse += Math.pow(prediction[j]-m_test[i][j], 2);
			}
			rmse /= prediction.length;
			rmse = Math.sqrt(rmse);
			avgRmse += rmse;
			
			logger.debug("Mean squared error: {}", rmse);
		}
		
		avgRmse /= x_test.length;
		logger.info("Total rmse {}", avgRmse);
		
		// basic performance assumption
		assertEquals(0.0, avgRmse, 20.0);
	}

	private double[][] getPortion(Instances data, int numClassifiers, boolean getClassifiers) {
		int numAttributes = getClassifiers ? numClassifiers : data.numAttributes() - numClassifiers;
		double[][] newData = new double[data.numInstances()][numAttributes];

		for (int i = 0; i < data.numInstances(); i++) {
			int index = 0;
			double[] instance = data.get(i).toDoubleArray();
			for (int j = 0; j < data.numAttributes(); j++) {
				if (data.attribute(j).name().startsWith("weka") == getClassifiers) {
					newData[i][index] = instance[j];
					index++;
				}
			}
		}

		return newData;
	}
}
