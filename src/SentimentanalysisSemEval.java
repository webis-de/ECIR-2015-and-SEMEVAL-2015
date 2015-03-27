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


public class SentimentanalysisSemEval {

	private Set<Tweet> tweetList = new HashSet<Tweet>();
	private String PATH =  "";
	
	public SentimentanalysisSemEval(String path) throws FileNotFoundException, UnsupportedEncodingException {
		this.PATH = path;
		loadTweets(path);
	}
	
	public void trainSystem(int system, String savename) throws IOException, ClassNotFoundException {
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
			case 3:
				SentimentSystemTeamX teamXSystem = new SentimentSystemTeamX(tweetList);
				teamXSystem.train(savename);
				break;
			default:
				throw new IllegalArgumentException("Invalid system: " + system);	
		}
	}
	public void trainAllSystems(int system, String savename) throws IOException {
			SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
			nrcSystem.train(savename);

			SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
			gumltltSystem.train(savename);

			SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
			klueSystem.train(savename);

//			TeamXSentimentSystem teamXSystem = new TeamXSentimentSystem(tweetList);
//			teamXSystem.train(savename);
	}
	
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
			case 3:
				SentimentSystemTeamX teamXSystem = new SentimentSystemTeamX(tweetList);
				this.evalModel(teamXSystem.test(trainname));
				break;
			default:
				throw new IllegalArgumentException("Invalid system: " + system);
		}
	}
	
	public void testAllSystem(String trainnameNRC, String trainnameGUMLTLT, String trainnameKLUE, String trainnameTeamX) throws Exception {
			SentimentSystemNRC nrcSystem = new SentimentSystemNRC(tweetList);
			Map<String, ClassificationResult> nrcResult = nrcSystem.test(trainnameNRC);

			SentimentSystemGUMLTLT gumltltSystem = new SentimentSystemGUMLTLT(tweetList);
			Map<String, ClassificationResult> gumltltResult = gumltltSystem.test(trainnameGUMLTLT);

			SentimentSystemKLUE klueSystem = new SentimentSystemKLUE(tweetList);
			Map<String, ClassificationResult> klueResult = klueSystem.test(trainnameKLUE);

			SentimentSystemTeamX teamXSystem = new SentimentSystemTeamX(tweetList);
			Map<String, ClassificationResult> teamxResult = teamXSystem.test(trainnameTeamX);
			
			this.evalAllModels(nrcResult, gumltltResult, klueResult, teamxResult);
	}
	
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
	
	private boolean storeTweetUni(String tweetString, String senti, String tweetID) throws UnsupportedEncodingException{
		Tweet tweet = new Tweet(tweetString, senti, tweetID);		
	    if(this.tweetList.add(tweet)){
	    	return true;
	    }
	    else {
			return false;
		}
    }
	
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
	
	private void evalAllModels(Map<String, ClassificationResult> nrcResult, Map<String, ClassificationResult> gumltltResult, Map<String, ClassificationResult> klueResult, Map<String, ClassificationResult> teamxResult) throws Exception {
		System.out.println("Starting print Pred");
		double[][] matrix = new double[3][3];
		Map<String, Integer> classValue = new HashMap<String, Integer>();
		classValue.put("positive", 0);
		classValue.put("neutral", 1);
		classValue.put("negative", 2);
		
//	    Map<Integer, String> classValue2 = new HashMap<Integer, String>();
//	        classValue2.put(0, "positive");
//	        classValue2.put(1, "neutral");
//	        classValue2.put(2, "negative");
		Map<String, Integer> resultMapToPrint = new HashMap<String, Integer>();
		if((nrcResult != null && gumltltResult != null && klueResult != null && teamxResult != null)  && (nrcResult.size() == gumltltResult.size()) && (nrcResult.size() == klueResult.size()) && (nrcResult.size() == teamxResult.size())){
			for (Map.Entry<String, ClassificationResult> tweet : nrcResult.entrySet()){
				String tweetID = tweet.getKey();
				ClassificationResult nRCSenti = tweet.getValue();
				ClassificationResult gUMLTLTSenti = gumltltResult.get(tweet.getKey());
				ClassificationResult kLUESenti = klueResult.get(tweet.getKey());
				ClassificationResult teamXSenti = teamxResult.get(tweet.getKey());
				if(gUMLTLTSenti != null && kLUESenti != null && teamXSenti != null){
					double[] useSentiArray = {0,0,0};
					for (int i = 0; i < 3; i++){
						useSentiArray[i] = (nRCSenti.getResultDistribution()[i] + gUMLTLTSenti.getResultDistribution()[i] + kLUESenti.getResultDistribution()[i] + teamXSenti.getResultDistribution()[i]) / 4;
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
//			              if (actualSenti == useSenti){
//			                    if ((nRCSenti.getResult() == 0 && useSenti == 2) || (nRCSenti.getResult() == 2 && useSenti == 0)){
//			                        System.out.println("NRC; " + nRCSenti.getResultAsString() + "; " + classValue2.get(useSenti) + "; " + tweet.getValue().getTweet().getRawTweetString()); 
//			                    }
//			                    
//		                      if ((gUMLTLTSenti.getResult() == 0 && useSenti == 2) || (gUMLTLTSenti.getResult() == 2 && useSenti == 0)){
//		                          System.out.println("GU-MLT-LT; " + gUMLTLTSenti.getResultAsString() + "; " + classValue2.get(useSenti) + "; " + tweet.getValue().getTweet().getRawTweetString()); 
//		                       }
//		                      
//		                      if ((kLUESenti.getResult() == 0 && useSenti == 2) || (kLUESenti.getResult() == 2 && useSenti == 0)){
//		                          System.out.println("KLUE; " + kLUESenti.getResultAsString() + "; " + classValue2.get(useSenti) + "; " + tweet.getValue().getTweet().getRawTweetString()); 
//		                       }
//			                }
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
		if (matrix.length != 0){
			System.out.println(matrix[0][0] +  " | " + matrix[0][1] + " | " + matrix[0][2]);
			System.out.println(matrix[1][0] +  " | " + matrix[1][1] + " | " + matrix[1][2]);
			System.out.println(matrix[2][0] +  " | " + matrix[2][1] + " | " + matrix[2][2]);
			score(matrix);
		}
		printResultToFile(resultMapToPrint);
	}
	
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
		System.out.println("f1: " + f1);
		System.out.println("f1 without neutral: " + (f1A + f1C) /2);
		System.out.println("precisionPos: " + precisionA);
		System.out.println("recallPos: " + recallA);
		System.out.println("precisionNeg: " + precisionC);
		System.out.println("recallNeg: " + recallC);
	}
	
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
