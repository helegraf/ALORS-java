package alors;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

import alors.latent_features.FeaturePredictorException;
import alors.matrix_completion.MatrixCompleterException;
import alors.matrix_completion.cofirank.CofiConfig;
import alors.matrix_completion.cofirank.CofirankCPlusPlus;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class AlorsTest {
	
	@AfterEach
	public void cleanUp() throws IOException {
		FileUtils.cleanDirectory(Paths.get("cofirank","config").toFile());
		FileUtils.deleteQuietly(Paths.get("cofirank","data","train.lsvm").toFile());
		FileUtils.deleteQuietly(Paths.get("cofirank","data","test.lsvm").toFile());
		FileUtils.cleanDirectory(Paths.get("cofirank","default_out").toFile());
	}

	@Test
	@DisabledOnOs(OS.WINDOWS)
	public void testALORSExecution() throws IOException, MatrixCompleterException, FeaturePredictorException{
		// read some instances
		BufferedReader reader = new BufferedReader(
				new FileReader(Paths.get("src", "test", "resources", "noProbing_nonan_noid.arff").toString()));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		Instances train = new Instances(data, 0, data.numInstances() - 10);
		Instances test = new Instances(data, data.numInstances() - 10, 10);

		// split into X (instance features) and M (instance w\ algorithm performances)
		double[][] x_train = getPortion(train,22,false);
		double[][] m_train = getPortion(train, 22, true);
		double[][] x_test = getPortion(test, 22, false);
		double[][] m_test = getPortion(test, 22, true);

		// train
		String executablePath = Paths.get("cofirank","dist","cofirank-deploy").toString();
		String configurationPath = Paths.get("cofirank","config","default.cfg").toString();
		String outFolderPath = Paths.get("cofirank","default_out").toString();
		String trainFilePath = Paths.get("cofirank","data","train.lsvm").toString();
		String testFilePath = Paths.get("cofirank","data","test.lsvm").toString();
		
		CofiConfig config = new CofiConfig(executablePath, configurationPath, outFolderPath, trainFilePath, testFilePath);
		CofirankCPlusPlus cofirank = new CofirankCPlusPlus(config);
		Alors alors = new Alors(cofirank);

		alors.completeMatrixAndPrepareColdStart(m_train, x_train);

		// test
		for (int i = 0; i < x_test.length; i++) {
			double[] prediction = alors.predictForFeatures(x_test[i]);
			System.out.println("prediction \t: " + Arrays.toString(prediction));
			System.out.println("real \t: " + Arrays.toString(m_test[i]));
		}
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
