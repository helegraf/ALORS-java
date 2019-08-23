package alors.matrix_completion.cofirank;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

import alors.matrix_completion.MatrixCompleterException;
import alors.matrix_completion.cofirank.CofiConfig;
import alors.matrix_completion.cofirank.CofirankCPlusPlus;

public class CofirankCPlusPlusTest {
	
	@AfterEach
	public void cleanUp() throws IOException {
		FileUtils.cleanDirectory(Paths.get("cofirank","config").toFile());
		FileUtils.cleanDirectory(Paths.get("cofirank","default_out").toFile());
	}

	@Test
	@DisabledOnOs(OS.WINDOWS)
	public void testComplete() throws MatrixCompleterException {
		String executablePath = Paths.get("cofirank","dist","cofirank-deploy").toString();
		String configurationPath = Paths.get("cofirank","config","default.cfg").toString();
		String outFolderPath = Paths.get("cofirank","default_out").toString();
		String trainFilePath = Paths.get("cofirank","data","dummytrain.lsvm").toString();
		String testFilePath = Paths.get("cofirank","data","dummytest.lsvm").toString();
		
		CofiConfig config = new CofiConfig(executablePath, configurationPath, outFolderPath, trainFilePath, testFilePath);
		CofirankCPlusPlus cofirank = new CofirankCPlusPlus(config);

		double nan = Double.NaN;
		double[][] matrix = { { 4, nan, 5, nan, 1, 2 }, { 5, 5, nan, nan, 2, 1 }, { nan, 4, 5, 1, nan, 1 },
				{ 1, 2, nan, nan, 4, 5 }, { 2, nan, 1, 5, 5, nan }, { nan, 1, 2, 4, nan, 5 } };
		double[][] completedMatrix = cofirank.complete(matrix);
		
		printMatrix(completedMatrix);
		
		double[][] u = cofirank.getU();
		printMatrix(u);
		
		double[][] v = cofirank.getV();
		printMatrix(v);
	}
	
	public void printMatrix(double[][] m) {
		for (int i = 0; i < m.length; i++) {
			System.out.println(Arrays.toString(m[i]));
		}
	}
}
