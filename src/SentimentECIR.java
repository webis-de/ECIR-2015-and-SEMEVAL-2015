import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class SentimentECIR {
	
	static ArrayList<String> stopWordList = new ArrayList<String>();
	static Set<Tweet> tweetList = new HashSet<Tweet>();
	static String mode;
	static String PATH = "";
	private static long startTime = System.currentTimeMillis();
	
	/**
	 * Main function
	 * @param args Command-Line Arguments
	 * @throws Exception
	 */	
	public static void main(String[] args) throws Exception{
		
		String nameOfNRCTrain = "";
		String nameOfGUMLTTrain = "";
		String nameOfKLUETrain = "";
		int evalmodelmode = 0;
		int trainmodelmode = 0;
		Options options = new Options();
		
		options.addOption("on", true, "output Name");
		options.addOption("tf", true, "Name of the NRC Trainfile");
		options.addOption("tf2", true, "Name of the GU-MLT-LT Trainfile");
		options.addOption("tf3", true, "Name of the KLUE Trainfile");
		options.addOption("em", true, "Eval Modelmode");
		options.addOption("tm", true, "Train Modelmode");
		
		CommandLineParser parser = new GnuParser();
		try {
			String name = "";
			CommandLine line = parser.parse(options, args);
			if(line.hasOption("on")){
				name = "_" + line.getOptionValue("on");
			}
			if(line.hasOption("tf")){
				nameOfNRCTrain = line.getOptionValue("tf");
			}
			if(line.hasOption("tf2")){
				nameOfGUMLTTrain = line.getOptionValue("tf2");
			}			
			if(line.hasOption("tf3")){
				nameOfKLUETrain = line.getOptionValue("tf3");
			}
			if(line.hasOption("em")){
				evalmodelmode = Integer.parseInt(line.getOptionValue("em"));
			}
			if(line.hasOption("tm")){
				trainmodelmode = Integer.parseInt(line.getOptionValue("tm"));
			}
			
			String[] argList = line.getArgs();
			PATH = argList[1]; //Path to Test or Traindata
			
			SentimentanalysisECIR sentimentanalysis = new SentimentanalysisECIR(PATH);
			
			switch (argList[0]){
				case "eval":
					sentimentanalysis.testSystem(evalmodelmode, nameOfNRCTrain);
					break;
				case "evalAll":
					sentimentanalysis.testAllSystem(nameOfNRCTrain, nameOfGUMLTTrain, nameOfKLUETrain);
					break;
				case "train":
					sentimentanalysis.trainSystem(trainmodelmode, name);
					break;
				case "trainAll":
					sentimentanalysis.trainAllSystems(name);
					break;
				default:
					throw new IllegalArgumentException("Invalid mode: " + argList[0]);
			}					
		}
		catch(ParseException exp){
			System.err.println("Parsing failed.  Reason: " + exp.getMessage());
		}
		long endTime = System.currentTimeMillis();
        System.out.println("It took " + ((endTime - startTime) / 1000) + " seconds");
				
	}
}	
