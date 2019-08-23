package alors;

import alors.latent_features.FeaturePredictor;
import alors.latent_features.FeaturePredictorException;
import alors.latent_features.WEKAFeaturePredictor;
import alors.matrix_completion.MatrixCompleterException;
import alors.matrix_completion.ModelBasedMatrixCompleter;

/**
 * A simple java implementation for ALORS [0], algorithm recommender system.
 * 
 * <p>[0] Mısır, Mustafa, and Michèle Sebag. "Alors: An algorithm recommender
 * system." Artificial Intelligence 244 (2017): 291-314.
 * 
 * @author helegraf
 *
 */
public class Alors {

	private ModelBasedMatrixCompleter matrixCompleter;
	private FeaturePredictor featurePredictor = new WEKAFeaturePredictor();

	/**
	 * Initialize Alors using the given matrix completer, which has to be model
	 * based as it has to produce latent features.
	 * 
	 * @param matrixCompleter the model-based matrix completer
	 */
	public Alors(ModelBasedMatrixCompleter matrixCompleter) {
		this.matrixCompleter = matrixCompleter;
	}

	/**
	 * Completes the given matrix as well as learning a mapping from instance to
	 * latent features.
	 * 
	 * @param matrixM (rows = instances (e.g. users/ datasets/ ...), columns = items
	 *                (e.g. movies/ algorithms/ ...)
	 * @param matrixX matrix of instance features (rows = instances, columns =
	 *                features)
	 * @return an estimated of a completed matrix m
	 * @throws MatrixCompleterException  if the matrix could not be completed
	 *                                   correctly or the latent features not learnt
	 * @throws FeaturePredictorException if the feature predictor for the latent
	 *                                   features could not be built
	 */
	public double[][] completeMatrixAndPrepareColdStart(double[][] matrixM, double[][] matrixX)
			throws MatrixCompleterException, FeaturePredictorException {
		// do matrix completion for M
		double[][] mHead = matrixCompleter.complete(matrixM);

		// train model for feature vector
		featurePredictor.train(matrixX, matrixCompleter.getU());

		return mHead;
	}

	/**
	 * Returns a prediction for the given instance.
	 * 
	 * @param featureVectorX the instance features
	 * @return a prediction of item values, e.g. algorithm performances or ranks for
	 *         the case of algorithm selection
	 * @throws FeaturePredictorException if the latent features for the instance
	 *                                   could not be predicted
	 */
	public double[] predictForFeatures(double[] featureVectorX) throws FeaturePredictorException {
		// feed into prediction model for rf; then multiply latent feature vector with
		// algorithm feature vector matrix
		double[] latentFeatures = featurePredictor.predict(featureVectorX);

		// TODO multiply with the predicted matrix!!
		return latentFeatures;
	}

	public ModelBasedMatrixCompleter getMatrixCompleter() {
		return matrixCompleter;
	}

	public void setMatrixCompleter(ModelBasedMatrixCompleter matrixCompleter) {
		this.matrixCompleter = matrixCompleter;
	}

	public FeaturePredictor getFeaturePredictor() {
		return featurePredictor;
	}

	public void setFeaturePredictor(FeaturePredictor featurePredictor) {
		this.featurePredictor = featurePredictor;
	}
}
