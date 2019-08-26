package alors.matrix_completion.cofirank;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import alors.matrix_completion.cofirank.CofiConfig;

/**
 * Tests {@link CofiConfig}.
 * 
 * @author helegraf
 *
 */
public class CofiConfigTest {

	/**
	 * Clean testing directory.
	 * 
	 * @throws IOException
	 */
	@AfterEach
	public void cleanUp() throws IOException {
		FileUtils.cleanDirectory(Paths.get("cofirank", "config").toFile());
	}

	/**
	 * Tests if {@link CofiConfig#createConfig()} creates a config (validity not
	 * ensured).
	 * 
	 * @throws IOException
	 */
	@Test
	public void testCreateConfig() throws IOException {
		String executablePath = Paths.get("cofirank", "dist", "cofirank-deploy").toString();
		String configurationPath = Paths.get("cofirank", "config", "default.cfg").toString();
		String outFolderPath = Paths.get("cofirank", "default_out").toString();
		String trainFilePath = Paths.get("cofirank", "data", "dummytrain.lsvm").toString();
		String testFilePath = Paths.get("cofirank", "data", "dummytest.lsvm").toString();

		CofiConfig config = new CofiConfig(executablePath, configurationPath, outFolderPath, trainFilePath,
				testFilePath);
		config.createConfig();

		assertEquals(true, Paths.get(config.getConfigurationPath()).toFile().exists());
	}
}
