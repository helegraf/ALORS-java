package alors.matrix_completion.cofirank;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import alors.matrix_completion.MatrixCompleterException;

/**
 * Tests {@link CofirankCPlusPlus}.
 * 
 * @author helegraf
 *
 */
public class CofirankCPlusPlusTest {
	
	private Logger logger = LoggerFactory.getLogger(CofirankCPlusPlusTest.class);
	
	/**
	 * Clean up all testing directories.
	 * 
	 * @throws IOException
	 */
	@AfterEach
	public void cleanUp() throws IOException {
		FileUtils.cleanDirectory(Paths.get("cofirank","config").toFile());
		FileUtils.cleanDirectory(Paths.get("cofirank","default_out").toFile());
	}

	/**
	 * Tests the functionality of cofirank for a very small problem. Only available under the OS for which cofirank has been compiled, default linux.
	 * 
	 * @throws MatrixCompleterException
	 */
	@Test
	@EnabledOnOs(OS.LINUX)
	public void testComplete() throws MatrixCompleterException {
		String executablePath = Paths.get("cofirank","dist","cofirank-deploy").toString();
		String configurationPath = Paths.get("cofirank","config","default.cfg").toString();
		String outFolderPath = Paths.get("cofirank","default_out").toString();
		String trainFilePath = Paths.get("cofirank","data","dummytrain.lsvm").toString();
		String testFilePath = Paths.get("cofirank","data","dummytest.lsvm").toString();
		
		int dimW = 10;
		CofiConfig config = new CofiConfig(executablePath, configurationPath, outFolderPath, trainFilePath, testFilePath);
		config.setDimW(dimW);
		CofirankCPlusPlus cofirank = new CofirankCPlusPlus(config);

		double nan = Double.NaN;
		double[][] matrix = { { 4, nan, 5, nan, 1, 2 }, { 5, 5, nan, nan, 2, 1 }, { nan, 4, 5, 1, nan, 1 },
				{ 1, 2, nan, nan, 4, 5 }, { 2, nan, 1, 5, 5, nan }, { nan, 1, 2, 4, nan, 5 } };
		double[][] completedMatrix = cofirank.complete(matrix);
		
		logger.info("M_head Matrix", matrixToString(completedMatrix));
		
		double[][] u = cofirank.getU();
		logger.info("U Matrix", matrixToString(u));
		
		double[][] v = cofirank.getV();
		logger.info("V Matrix", matrixToString(v));
		
		// Check dimensions of results for basic working check
		assertEquals(matrix.length,completedMatrix.length);
		assertEquals(matrix[0].length, completedMatrix[0].length);
		
		assertEquals(dimW, u[0].length);
		assertEquals(dimW, v[0].length);
		
		assertEquals(matrix.length, u.length);
		assertEquals(matrix.length, v.length);
	}
	
	private String matrixToString(double[][] m) {
		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < m.length; i++) {
			buffer.append(Arrays.toString(m[i]));
			buffer.append(System.lineSeparator());
		}
		return buffer.toString();
	}
}
