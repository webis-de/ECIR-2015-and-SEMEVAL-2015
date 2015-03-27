import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.functions.LibLINEAR;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import cmu.arktweetnlp.Tagger;

/**
 * Trains and tests the KLUE system
 */
public class SentimentSystemKLUE extends SentimentSystem {
    
    /**
     * Constructor gets all Tweets in a list.
     * 
     * @param tweetList the list with all Tweets.
     */
	public SentimentSystemKLUE(Set<Tweet> tweetList) {
		super(tweetList);
	}
	
	/**
	 * Creates all features and instances for the trainingdata and saves them in an arff file
	 * 
	 * @param savename optional filename for the arff file
	 * @throws IOException
	 */
	public void train(String savename) throws IOException{
		System.out.println("Starting KLUE Train");
		System.out.println("Tweets: " +  tweetList.size());
		
		//load pos-tagger
    	Tagger tagger = new Tagger();
    	tagger.loadModel("resources/tagger/model.20120919.txt");
    	
    	//load sentiment lexica
    	Map<String, Double> afinnLexi = this.loadAFINN();
		
		int featurecount = 0;
		Map<String, Integer> nGramMap = new HashMap<String, Integer>();
		ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
		
		//creating features
		for(Tweet tweet : tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.tokenizeAndTag(tagger, tweet);
			this.negate(tweet);
			
			//get n-grams and set n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 2);
			for (String nGram : nGramSet){
				if(!nGramMap.containsKey(nGram)){
					nGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("NGRAM_" + nGram));
				}
			}
		}
		
		//set lexica features
		Attribute sentiAFINNPos = new Attribute("sentiAFINNPos");
		attributeList.add(sentiAFINNPos);
		featurecount++;
		Attribute sentiAFINNNeg = new Attribute("sentiAFINNNeg");
		attributeList.add(sentiAFINNNeg);
		featurecount++;
		Attribute sentiAFINNTotal = new Attribute("sentiAFINNTotal");
		attributeList.add(sentiAFINNTotal);
		featurecount++;
		Attribute sentiAFINNScore = new Attribute("sentiAFINNScore");
		attributeList.add(sentiAFINNScore);
		featurecount++;
		
		Attribute sentiEmoPos = new Attribute("sentiEmoPos");
		attributeList.add(sentiEmoPos);
		featurecount++;
		Attribute sentiEmoNeg = new Attribute("sentiEmoNeg");
		attributeList.add(sentiEmoNeg);
		featurecount++;
		Attribute sentiEmoTotal = new Attribute("sentiEmoTotal");
		attributeList.add(sentiEmoTotal);
		featurecount++;
		Attribute sentiEmoScore = new Attribute("sentiEmoScore");
		attributeList.add(sentiEmoScore);
		featurecount++;
		
		Attribute tokenCount = new Attribute("tokenCount");
		attributeList.add(tokenCount);
		featurecount++;
		
		//set class attribute
	    ArrayList<String> fvClassVal = new ArrayList<String>();
	    fvClassVal.add("positive");
	    fvClassVal.add("neutral");
	    fvClassVal.add("negative");
	    Attribute classAttribute = new Attribute("Class", fvClassVal);
	    attributeList.add(classAttribute);
		featurecount++;
		
		//creating instances with features
		Instances trainingSet = new Instances("test", attributeList, tweetList.size());
		trainingSet.setClassIndex(classAttribute.index());
		
		for(Tweet tweet : tweetList){
			SparseInstance instance = new SparseInstance(0);
			
			//n-gram feature
			Set<String> nGramSet = tweet.getnGramList();
			for (String nGram : nGramSet){
				Integer index = nGramMap.get(nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//lexica features
			List<Double> afinnScore = this.getAFINNScore(afinnLexi, tweet.getStemList());
			instance.setValue(sentiAFINNPos, afinnScore.get(0));
			instance.setValue(sentiAFINNNeg, afinnScore.get(1));
			instance.setValue(sentiAFINNTotal, afinnScore.get(2));
			instance.setValue(sentiAFINNScore, afinnScore.get(3));
			
			List<Double> emoScore = this.getEmoScore(tweet);
			instance.setValue(sentiEmoPos, emoScore.get(0));
			instance.setValue(sentiEmoNeg, emoScore.get(1));
			instance.setValue(sentiEmoTotal, emoScore.get(2));
			instance.setValue(sentiEmoScore, emoScore.get(3));
			
			instance.setValue(tokenCount, tweet.getWordList().size());
			
			//set class attribute
			instance.setValue(classAttribute, tweet.getSentiment());
			
			trainingSet.add(instance);
		}
		
		//save features and training instances in .arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingSet);
		saver.setFile(new File("resources/arff/Trained-Features-KLUE" + savename + ".arff"));
		saver.writeBatch();
		System.out.println("Trained-Features-KLUE" + savename + " saved");
	}
	
	/**
	 * Creates all features and instances for the testdata and classifies the Tweet
	 * 
	 * @param nameOfTrain optional filename of the arff file
	 * @return returns all results in a map
	 * @throws Exception
	 */
	public Map<String,ClassificationResult> test(String nameOfTrain) throws Exception{
		System.out.println("Starting KLUE Test");
		System.out.println("Tweets: " +  this.tweetList.size());
		String trainname = "";
		if(!nameOfTrain.equals("")){
			trainname = nameOfTrain;
		}
		else{
			trainname = "Trained-Features-KLUE";
		}
		
		//load features and training instances from .arff file
		BufferedReader reader = new BufferedReader(new FileReader("resources/arff/" + trainname + ".arff"));
		Instances train = new Instances(reader);
		train.setClassIndex(train.numAttributes() - 1);
		reader.close();
		
		//load and setup classifier
		LibLINEAR classifier = new LibLINEAR();
        classifier.setProbabilityEstimates(true);
        classifier.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
		classifier.setCost(0.05);
		
		//train classifier with instances
		classifier.buildClassifier(train);
		
		//delete train instances, to use same features with test instances
		train.delete();
		
		//load pos-tagger
        Tagger tagger = new Tagger();
        tagger.loadModel("resources/tagger/model.20120919.txt");
		
        //load sentiment lexica
    	Map<String, Double> afinnLexi = this.loadAFINN();
    	
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		for (int i = 0; i < train.numAttributes(); i++){
			featureMap.put(train.attribute(i).name(), train.attribute(i).index());
		}
		
		Map<String, ClassificationResult> resultMap = new HashMap<String, ClassificationResult>();
		for(Tweet tweet : tweetList){
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.tokenizeAndTag(tagger, tweet);
			this.negate(tweet);
			SparseInstance instance = new SparseInstance(0);
			
			//creating test instances with features
			//n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 2);
			for (String nGram : nGramSet){
				Integer index = featureMap.get("NGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//lexica features
			List<Double> afinnScore = this.getAFINNScore(afinnLexi, tweet.getStemList());
			instance.setValue(featureMap.get("sentiAFINNPos"), afinnScore.get(0));
			instance.setValue(featureMap.get("sentiAFINNNeg"), afinnScore.get(1));
			instance.setValue(featureMap.get("sentiAFINNTotal"), afinnScore.get(2));
			instance.setValue(featureMap.get("sentiAFINNScore"), afinnScore.get(3));
			List<Double> emoScore = this.getEmoScore(tweet);
			instance.setValue(featureMap.get("sentiEmoPos"), emoScore.get(0));
			instance.setValue(featureMap.get("sentiEmoNeg"), emoScore.get(1));
			instance.setValue(featureMap.get("sentiEmoTotal"), emoScore.get(2));
			instance.setValue(featureMap.get("sentiEmoScore"), emoScore.get(3));
			
			instance.setValue(featureMap.get("tokenCount"), tweet.getWordList().size());

			//add test instance to trained features
			train.add(instance);
			
			//classify Tweet
			double result = classifier.classifyInstance(train.lastInstance());
			double[] resultDistribution = classifier.distributionForInstance(train.lastInstance());
			resultMap.put(tweet.getTweetID(), new ClassificationResult(tweet, resultDistribution, result));
		}

		return resultMap;
	}
	
	//helper functions to preprocess and get features
	
	/**
	 * Preprocesses the Tweet
	 * 
	 * @param tweet the raw Tweet
	 */
	private void preProcessTweet(Tweet tweet){
		String rawTweet = tweet.getRawTweetString();
		rawTweet = rawTweet.toLowerCase();
		//filter Usernames
		rawTweet = rawTweet.replaceAll("@[^\\s]+", "");
		//filter Urls
		rawTweet = rawTweet.replaceAll("((www\\.[^\\s]+)|(https?://[^\\s]+))", "");
		tweet.setTweetString(rawTweet.trim());
	}
	
	/**
	 * Tokenize and pos-taggs the Tweet
	 * 
	 * @param tagger the ARK PoS-Tagger
	 * @param tweet the Tweet to tag
	 * @throws IOException
	 */
	private void tokenizeAndTag(Tagger tagger, Tweet tweet) throws IOException{
    	tweet.setWordList(tagger.tokenizeAndTag(tweet.getTweetString()));
    	this.getStems(tweet);
    }
	
	/**
	 * Calculates the AFINN scores for the lexica feature
	 * 
	 * @param map a map with all the lexica scores
	 * @param wordList a list of words that occur in the Tweet
	 * @return returns a list with all lexica features for the AFINN lexica
	 */
	private List<Double> getAFINNScore(Map<String,Double> map, Set<String> wordList) {
    	double posCount = 0.0;
    	double negCount = 0.0;
    	double totalCount = 0.0;
    	double totalScore = 0.0;
    	for (String word : wordList){
    		Double score = map.get(word);
    		if(score != null){
    			totalScore = totalScore + score;
    			if (score > 0){
    				posCount++;
    			}
    			if (score < 0){
    				negCount++;
    			}
    			totalCount++;
    		}
    	}
    	List<Double> scoreList = new ArrayList<Double>();
    	scoreList.add(posCount);
    	scoreList.add(negCount);
    	scoreList.add(totalCount);
    	scoreList.add(totalScore);
    	return scoreList;
	}
	
	/**
	 * Calculates the emoticon scores for the emoticon feature
	 * 
	 * @param tweet the Tweet to analyze
	 * @return returns a list with all emoticon features
	 */
	private List<Double> getEmoScore(Tweet tweet) {
		double posCount = 0.0;
		double negCount = 0.0;
		double totalCount = 0.0;
		double totalScore = 0.0;
		Set<String> emoticons =  this.getEmoticons(tweet);
		for (String emo : emoticons){
			totalCount++;
			if(emo.endsWith("(") || emo.endsWith("[") || emo.endsWith("<") || emo.endsWith("/") || emo.toLowerCase().endsWith("c") || emo.startsWith(")") || emo.startsWith("]") || emo.startsWith(">") || emo.startsWith("\\") || emo.startsWith("D")){
				negCount++;
				totalScore = totalScore - 1;
			}
			if(emo.endsWith(")") || emo.endsWith("]") || emo.endsWith(">") || emo.endsWith("D") || emo.startsWith("(") || emo.startsWith("[") || emo.startsWith("<") || emo.toLowerCase().startsWith("c")){    			
				posCount++;
				totalScore = totalScore + 1;
			}
		}
		List<Double> scoreList = new ArrayList<Double>();
		scoreList.add(posCount);
		scoreList.add(negCount);
		scoreList.add(totalCount);
		scoreList.add(totalScore);
		return scoreList;
	}
	
}
