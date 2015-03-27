import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.lang.StringUtils;

/**
 * Handels traning, testing and evaluation of the Sentimentsystems.
 */
public class SentimentanalysisECIR {

	private Set<Tweet> tweetList = new HashSet<Tweet>();
	private String PATH =  "";
	
	/**
	 * Constructor loads all Tweets from a Path.
	 * 
	 * @param path the path to the Tweetfile.
	 * @throws FileNotFoundException
	 * @throws UnsupportedEncodingException
	 */
	public SentimentanalysisECIR(String path) throws FileNotFoundException, UnsupportedEncodingException {
		this.PATH = path; //path to train or test file
		loadTweets(path);
	}
	
	/**
	 * Trains a specific system
	 * 
	 * @param system system ID (0: NRC; 1: GU-MLT-LT; 2: KLUE)
	 * @param savename optional filename for the arff file
	 * @throws IOException
	 */
	public void trainSystem(int system, String savename) throws IOException {
		switch (system){
			case 0:
				SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
				nrcSystem.train(savename);
				break;
			case 1:
				SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
				gumltltSystem.train(savename);
				break;
			case 2:	
				SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
				klueSystem.train(savename);
				break;
			default:
				throw new IllegalArgumentException("Invalid system: " + system);	
		}
	}
	
	/**
	 * Trains all 3 systems
	 * 
	 * @param savename optional filename for the arff file
	 * @throws IOException
	 */
	public void trainAllSystems(String savename) throws IOException {
			SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
			nrcSystem.train(savename);

			SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
			gumltltSystem.train(savename);

			SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
			klueSystem.train(savename);
	}
	
	/**
	 * Tests and evaluate a specific system
	 * 
	 * @param system system ID (0: NRC; 1: GU-MLT-LT; 2: KLUE)
	 * @param trainname optional filename of the arff file
	 * @throws Exception
	 */
	public void testSystem(int system, String trainname) throws Exception {
		switch (system){
			case 0:
				SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
				this.evalModel(nrcSystem.test(trainname));
				break;
			case 1:
				SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
				this.evalModel(gumltltSystem.test(trainname));
				break;
			case 2:
				SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
				this.evalModel(klueSystem.test(trainname));
				break;
			default:
				throw new IllegalArgumentException("Invalid system: " + system);
		}
	}
	
	
	/**
	 * Tests and evaluate all 3 systems
	 * 
	 * @param trainnameNRC optional filename of the arff file for the NRC system
	 * @param trainnameGUMLTLT optional filename of the arff file for the GU-MLT-LT system
	 * @param trainnameKLUE optional filename of the arff file for the KLUE system
	 * @throws Exception
	 */
	public void testAllSystem(String trainnameNRC, String trainnameGUMLTLT, String trainnameKLUE) throws Exception {
			SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
			Map<String, ClassificationResult> nrcResult = nrcSystem.test(trainnameNRC);

			SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
			Map<String, ClassificationResult> gumltltResult = gumltltSystem.test(trainnameGUMLTLT);

			SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
			Map<String, ClassificationResult> klueResult = klueSystem.test(trainnameKLUE);

			
			this.evalAllModels(nrcResult, gumltltResult, klueResult);
	}
	
	/**
	 * Parse Tweets from train or test file
	 * 
	 * @param path path to train or test file
	 * @throws FileNotFoundException
	 * @throws UnsupportedEncodingException
	 */
	private void loadTweets(String path) throws FileNotFoundException, UnsupportedEncodingException{
		File file = new File("resources/tweets/" + path + ".txt");
		Scanner scanner = new Scanner(file);
		int multiple = 0;
		while (scanner.hasNextLine()) {
			String[] line = scanner.nextLine().split("\t");
			if (line.length == 4){
				if (line[0].equals("NA")){
					if (!storeTweetUni(line[3], line[2], line[1])){
						System.out.println("Tweet already in list: " + line[1]);
						multiple++;
					}
				}
				else{
					if (!storeTweetUni(line[3], line[2], line[0])){
						System.out.println("Tweet already in list: " + line[0]);
						multiple++;
					}
				}
			}
			else{
			    System.out.println("Wrong format: " + line[0]);
			}
		}
		System.out.println("multiple Tweets: " + multiple);
		scanner.close();
	}
	
	/**
	 * Stores Tweet in tweetList, if not already in there
	 * 
	 * @param tweetString the Tweetstring
	 * @param senti the Tweet Sentiment
	 * @param tweetID the Tweet ID
	 * @return true if the Tweet was added to the list, false if the Tweet was already in the list
	 * @throws UnsupportedEncodingException
	 */
	private boolean storeTweetUni(String tweetString, String senti, String tweetID) throws UnsupportedEncodingException{
		Tweet tweet = new Tweet(tweetString, senti, tweetID);		
	    if(this.tweetList.add(tweet)){
	    	return true;
	    }
	    else {
			return false;
		}
    }
	
	/**
	 * Evaluate a specific system
	 * 
	 * @param resultMap a map with all classified Tweets
	 * @throws Exception
	 */
	private void evalModel(Map<String, ClassificationResult> resultMap) throws Exception {
		System.out.println("Starting eval Model");
		System.out.println("Tweets: " +  tweetList.size());
		double[][] matrix = new double[3][3];
		Map<String, Integer> classValue = new HashMap<String, Integer>();
		classValue.put("positive", 0);
		classValue.put("neutral", 1);
		classValue.put("negative", 2);
		Map<String, Integer> resultMapToPrint = new HashMap<String, Integer>();
		for (Map.Entry<String, ClassificationResult> tweet : resultMap.entrySet()){
			String tweetID = tweet.getKey();
			ClassificationResult senti = tweet.getValue();
			double[] useSentiArray = {0,0,0};
			for (int i = 0; i < 3; i++){
				useSentiArray[i] = (senti.getResultDistribution()[i]);
			}
			int useSenti = 1;
			if(useSentiArray[0] > useSentiArray[1] && useSentiArray[0] > useSentiArray[2]){
				useSenti = 0;
			}
			if(useSentiArray[2] > useSentiArray[0] && useSentiArray[2] > useSentiArray[1]){
				useSenti = 2;
			}
			resultMapToPrint.put(tweetID, useSenti);
			if (!tweet.getValue().getTweet().getSentiment().equals("unknwn")){
				Integer actualSenti = classValue.get(tweet.getValue().getTweet().getSentiment());
				matrix[actualSenti][useSenti]++;
			}
		}
		if (matrix.length != 0){
			System.out.println(matrix[0][0] +  " | " + matrix[0][1] + " | " + matrix[0][2]);
			System.out.println(matrix[1][0] +  " | " + matrix[1][1] + " | " + matrix[1][2]);
			System.out.println(matrix[2][0] +  " | " + matrix[2][1] + " | " + matrix[2][2]);
			score(matrix);
		}
		printResultToFile(resultMapToPrint);
	}
	
	/**
	 * Evaluate the ensemble system
	 * 
	 * @param nrcResult a map with all classified Tweets form the NRC system
	 * @param gumltltResult a map with all classified Tweets form the GU-MLT-LT system
	 * @param klueResult a map with all classified Tweets form the KLUE system
	 * @throws Exception
	 */
	private void evalAllModels(Map<String, ClassificationResult> nrcResult, Map<String, ClassificationResult> gumltltResult, Map<String, ClassificationResult> klueResult) throws Exception {
		System.out.println("Starting print Pred");
		double[][] matrix = new double[3][3];
		Map<String, Integer> classValue = new HashMap<String, Integer>();
		classValue.put("positive", 0);
		classValue.put("neutral", 1);
		classValue.put("negative", 2);
		
		Map<String, Integer> resultMapToPrint = new HashMap<String, Integer>();
		if((nrcResult != null && gumltltResult != null && klueResult != null)  && (nrcResult.size() == gumltltResult.size()) && (nrcResult.size() == klueResult.size())){
			for (Map.Entry<String, ClassificationResult> tweet : nrcResult.entrySet()){
				String tweetID = tweet.getKey();
				ClassificationResult nRCSenti = tweet.getValue();
				ClassificationResult gUMLTLTSenti = gumltltResult.get(tweet.getKey());
				ClassificationResult kLUESenti = klueResult.get(tweet.getKey());
				if(gUMLTLTSenti != null && kLUESenti != null){
					double[] useSentiArray = {0,0,0};
					for (int i = 0; i < 3; i++){
					    //use average of all distributions as end Sentiment
						useSentiArray[i] = (nRCSenti.getResultDistribution()[i] + gUMLTLTSenti.getResultDistribution()[i] + kLUESenti.getResultDistribution()[i]) / 3;
					}
					int useSenti = 1;
					if(useSentiArray[0] > useSentiArray[1] && useSentiArray[0] > useSentiArray[2]){
						useSenti = 0;
					}
					if(useSentiArray[2] > useSentiArray[0] && useSentiArray[2] > useSentiArray[1]){
						useSenti = 2;
					}
					
					//store result
					resultMapToPrint.put(tweetID, useSenti);
					
					//store result in confusion matrix, if test data include GOLD standards
					if (!tweet.getValue().getTweet().getSentiment().equals("unknwn")){
						Integer actualSenti = classValue.get(tweet.getValue().getTweet().getSentiment());
						matrix[actualSenti][useSenti]++;
					}
				}
				else{
					System.out.println(tweet.getValue().getTweet().getTweetString());
				}
			}
		}
		else{
			System.out.println("resultMaps null or diffrent size");
		}
		
		//print confusion matrix and calculate F1 score
		if (matrix.length != 0){
			System.out.println(matrix[0][0] +  " | " + matrix[0][1] + " | " + matrix[0][2]);
			System.out.println(matrix[1][0] +  " | " + matrix[1][1] + " | " + matrix[1][2]);
			System.out.println(matrix[2][0] +  " | " + matrix[2][1] + " | " + matrix[2][2]);
			score(matrix);
		}
		
		//print result to file
		printResultToFile(resultMapToPrint);
	}
	
	/**
	 * Calculates the F1 Score
	 * 
	 * @param matrix the confusion matrix
	 */
	private void score(double[][] matrix){
		double precisionA = matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0]);
		double precisionB = matrix[1][1] / (matrix[1][1] + matrix[2][1] + matrix[0][1]);
		double precisionC = matrix[2][2] / (matrix[2][2] + matrix[0][2] + matrix[1][2]);

		double precision = (precisionA + precisionB + precisionC) / 3;
		
		double recallA = matrix[0][0] / (matrix[0][0] + matrix[0][1] + matrix[0][2]);
		double recallB = matrix[1][1] / (matrix[1][1] + matrix[1][2] + matrix[1][0]);
		double recallC = matrix[2][2] / (matrix[2][2] + matrix[2][0] + matrix[2][1]);
		double recall = (recallA + recallB + recallC) / 3;
		
		double f1 = 2 * ((precision * recall) / (precision + recall));
		double f1A = 2 * ((precisionA * recallA) / (precisionA + recallA));
//		double f1B = 2 * ((precisionB * recallB) / (precisionB + recallB));
		double f1C = 2 * ((precisionC * recallC) / (precisionC + recallC));
		
		System.out.println("precision: " + precision);
		System.out.println("recall: " + recall);
//	    System.out.println("precisionPos: " + precisionA);
//	    System.out.println("recallPos: " + recallA);
//	    System.out.println("precisionNeg: " + precisionC);
//	    System.out.println("recallNeg: " + recallC);
		System.out.println("f1: " + f1);
		System.out.println("f1 without neutral: " + (f1A + f1C) / 2);

	}
	
	/**
	 * Prints the result of the sentiment analysis to the result file
	 * 
	 * @param resultMapToPrint a map with the results for all Tweets
	 * @throws FileNotFoundException
	 */	
	private void printResultToFile(Map<String, Integer> resultMapToPrint) throws FileNotFoundException {    
        int errorcount = 0;
        Map<Integer, String> classValue = new HashMap<Integer, String>();
        classValue.put(0, "positive");
        classValue.put(1, "neutral");
        classValue.put(2, "negative");
        File file = new File("resources/tweets/" + this.PATH + ".txt");
        PrintStream tweetPrintStream = new PrintStream(new File("resources/erg/result.txt"));
        Scanner scanner = new Scanner(file);
        while (scanner.hasNextLine()) {
            String[] line = scanner.nextLine().split("\t");
            String id = line[0];
            if (line[0].equals("NA")){
            	id = line[1];
            }
            if (line.length == 4){        
                String senti = classValue.get(resultMapToPrint.get(id));
                if (senti != null){
                    line[2] = senti;
                }
                else{
                    System.out.println("Error while printResultToFile: tweetID:" + id);
                    errorcount++;
                    line[2] = "neutral";
                }
            }
            else{
                errorcount++;
                System.out.println(line[0]);
            }           
            tweetPrintStream.print(StringUtils.join(line, "\t"));
            tweetPrintStream.println();
        }
        scanner.close();
        tweetPrintStream.close();
        if (errorcount != 0) System.out.println("Errors while printResultToFile: " + errorcount);
	}
}
