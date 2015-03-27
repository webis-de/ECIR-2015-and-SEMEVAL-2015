import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.classifiers.functions.LibLINEAR;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Tagger.TaggedToken;
import cmu.arktweetnlp.impl.features.WordClusterPaths;

/**
 * Trains and tests the GU-MLT-LT system
 */
public class SentimentSystemGUMLTLT extends SentimentSystem {

    /**
     * Constructor gets all Tweets in a list.
     * 
     * @param tweetList the list with all Tweets.
     */
	public SentimentSystemGUMLTLT(Set<Tweet> tweetList)  {
		super(tweetList);
	}

	/**
     * Creates all features and instances for the trainingdata and saves them in an arff file
     * 
     * @param savename optional filename for the arff file
     * @throws IOException
     */
	public void train(String savename) throws IOException{
		System.out.println("Starting GU-MLT-LT Train");
		System.out.println("Tweets: " +  tweetList.size());
		
		//load pos-tagger
    	Tagger tagger = new Tagger();
    	tagger.loadModel("resources/tagger/model.20120919.txt");
    	
    	//load sentiment lexica
    	Map<String, Double> sentiWordNet = this.loadSentiWordNet(); 

		int featurecount = 0;
		Map<String, Integer> nGramMap = new HashMap<String, Integer>();
		Map<String, Integer> stemMap = new HashMap<String, Integer>();
		Map<String, Integer> clusterMap = new HashMap<String, Integer>();
		ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
		
		//creating features
		for(Tweet tweet : tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.tokenizeAndTag(tagger, tweet);
			this.negate(tweet);
			
			//get n-grams and set n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 1);
			for (String nGram : nGramSet){
				if(!nGramMap.containsKey(nGram)){
					nGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("NGRAM_" + nGram));
				}
			}
			
			//get stems and set stem feature
			Set<String> stemSet = this.getStems(tweet);
			for (String stem : stemSet){
				if(!stemMap.containsKey(stem)){
					stemMap.put(stem, featurecount++);
					attributeList.add(new Attribute("STEM_" + stem));
				}
			}
			
			//get cluster and set cluster feature
			Set<String> clusterSet = this.getGUMLTLTClusters(tweet);
			for(String cluster : clusterSet){
				if(!clusterMap.containsKey(cluster)){
					clusterMap.put(cluster, featurecount++);
					attributeList.add(new Attribute("CLUSTER_" + cluster));
				}
			}
		}
		
		//set lexica features
		Attribute sentiWordNetPos = new Attribute("sentiWordNetPos");
		attributeList.add(sentiWordNetPos);
		featurecount++;
		Attribute sentiWordNetNeg = new Attribute("sentiWordNetNeg");
		attributeList.add(sentiWordNetNeg);
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
			
			//stem feature
			Set<String> stemSet = tweet.getStemList();
			for (String stem : stemSet){
				Integer index = stemMap.get(stem);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//cluster feature
			Set<String> clusterSet = tweet.getClusterList();
			for(String cluster : clusterSet){
				Integer index = clusterMap.get(cluster);
				if(index != null){
					instance.setValue(index, 1);
				}
			}

			//lexica features
			instance.setValue(sentiWordNetPos, this.getSentiWordNetScore("+", sentiWordNet, tweet.getCollapsedWordList()));
			instance.setValue(sentiWordNetNeg, this.getSentiWordNetScore("-", sentiWordNet, tweet.getCollapsedWordList()));
			
			//set class attribute
			instance.setValue(classAttribute, tweet.getSentiment());
			
			trainingSet.add(instance);
		}
		
		//save features and training instances in .arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingSet);
		saver.setFile(new File("resources/arff/Trained-Features-GUMLTLT" + savename + ".arff"));
		saver.writeBatch();
		System.out.println("Trained-Features-GUMLTLT" + savename + " saved");
	}
	
    /**
     * Creates all features and instances for the testdata and classifies the Tweet
     * 
     * @param nameOfTrain optional filename of the arff file
     * @return returns all results in a map
     * @throws Exception
     */	
	public Map<String, ClassificationResult> test(String nameOfTrain) throws Exception{
		System.out.println("Starting GUMLTLT Test");
		System.out.println("Tweets: " +  this.tweetList.size());
		String trainname = "";
		if(!nameOfTrain.equals("")){
			trainname = nameOfTrain;
		}
		else{
			trainname = "Trained-Features-GUMLTLT";
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
		classifier.setCost(0.15);
		
		//train classifier with instances
		classifier.buildClassifier(train);
		
		//delete train instances, to use same features with test instances
		train.delete();
		
		//load pos-tagger		
        Tagger tagger = new Tagger();
        tagger.loadModel("resources/tagger/model.20120919.txt");
		
        //load sentiment lexica
    	Map<String, Double> sentiWordNet = this.loadSentiWordNet();
    	
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
			Set<String> nGramSet = this.getNGrams(tweet, 1);
			for (String nGram : nGramSet){
				Integer index = featureMap.get("NGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//stem feature
			Set<String> stemSet = this.getStems(tweet);
			for (String stem : stemSet){
				Integer index = featureMap.get("STEM_" + stem);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//cluster feature
			Set<String> clusterSet = this.getGUMLTLTClusters(tweet);
			for(String cluster : clusterSet){
				Integer index = featureMap.get("CLUSTER_" + cluster);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//lexica features
			instance.setValue(featureMap.get("sentiWordNetPos"), this.getSentiWordNetScore("+", sentiWordNet, tweet.getCollapsedWordList()));
			instance.setValue(featureMap.get("sentiWordNetNeg"), this.getSentiWordNetScore("-", sentiWordNet, tweet.getCollapsedWordList()));

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
    	tweet.setRawWordList(tagger.tokenizeAndTag(tweet.getRawTweetString()));
    	tweet.setWordList(tagger.tokenizeAndTag(tweet.getTweetString()));
    	tweet.setCollapseList(this.collapseTweet(tweet));
    }
    
    /**
     * Collapses the Tweet
     * 
     * @param tweet the Tweet to analyze
     * @return the collapsed Tweet as a list of TaggedToken
     */
	private List<TaggedToken> collapseTweet(Tweet tweet){
		List<TaggedToken> tokenList = new ArrayList<TaggedToken>(tweet.getWordList());
		for (TaggedToken token : tokenList){
			Matcher matchWord = Pattern.compile("(.)\\1{2,}").matcher(token.token);
			String tempWord = token.token;
			while (matchWord.find()) {
				if(matchWord.end() < 0 || matchWord.start() < 0){
					
				}
				token.token = tempWord.substring(0, matchWord.start()+2) + tempWord.substring(matchWord.end());
			}
    	}
		return tokenList;
	}
	
	
	/**
	 * Determine the Cluster IDs for all 3 preprocessed versions of the Tweet 
	 * 
	 * @param tweet the Tweet to analyze
	 * @return returns a set of Cluster IDs
	 */
    private Set<String> getGUMLTLTClusters(Tweet tweet) {
    	Set<String> clusters = this.getGUMLTLTClusters(tweet.getWordList());
    	clusters.addAll(this.getGUMLTLTClusters(tweet.getRawWordList()));
    	clusters.addAll(this.getGUMLTLTClusters(tweet.getCollapsedWordList()));
    	tweet.setClusters(clusters);
    	return clusters;
	}
    
    
    /**
     * Determine the Cluster IDs for a preprocessed version of the Tweet 
     * 
     * @param wordList a list of words that occur in the Tweet
     * @return returns a set of Cluster IDs
     */
    private Set<String> getGUMLTLTClusters(List<TaggedToken> wordList) {
    	Set<String> clusterList = new HashSet<String>();
    	for (TaggedToken token : wordList){
    		String cluster = WordClusterPaths.wordToPath.get(token.token);
    		if (cluster != null){
    			clusterList.add(cluster);
    		}
    	}
		return clusterList;
    }
    
    
    /**
     * Calculates the SentiWordNet score for the lexica feature
     * 
     * @param posNeg get positive or negative scores
     * @param map a map with all the lexica scores
     * @param wordList a list of words that occur in the Tweet
     * @return returns the SentiWordNet score for the lexica feature
     */
	private Double getSentiWordNetScore(String posNeg, Map<String,Double> map, List<TaggedToken> wordList) {
    	double totalScore = 0.0;
    	for (TaggedToken token : wordList){
    		Double score = map.get(token.token + posNeg);
    		if(score != null){
    			totalScore = totalScore + score;
    		}
    	}
    	return totalScore;
	}
}
