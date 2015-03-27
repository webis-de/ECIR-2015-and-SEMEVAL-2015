import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
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

/**
 * Trains and tests the NRC system
 */
public class SentimentSystemNRC extends SentimentSystem {

    /**
     * Constructor gets all Tweets in a list.
     * 
     * @param tweetList the list with all Tweets.
     */    
	public SentimentSystemNRC(Set<Tweet> tweetList) {
		super(tweetList);
	}

    /**
     * Creates all features and instances for the trainingdata and saves them in an arff file
     * 
     * @param savename optional filename for the arff file
     * @throws IOException
     */	
	public void train(String savename) throws IOException{
		System.out.println("Starting NRC Train");
		System.out.println("Tweets: " +  this.tweetList.size());
		
		//load pos-tagger
    	Tagger tagger = new Tagger();
    	tagger.loadModel("resources/tagger/model.20120919.txt");
    	
    	//load sentiment lexica
    	Map<String, Double> senti140UniLexi = this.loadLexicon("sentiment140/unigrams-pmilexicon");
    	Map<String, Double> hashtagUniLexi = this.loadLexicon("hashtag/unigrams-pmilexicon");
    	Map<String, Double> senti140BiLexi = this.loadLexicon("sentiment140/bigrams-pmilexicon");
    	Map<String, Double> hashtagBiLexi = this.loadLexicon("hashtag/bigrams-pmilexicon");
    	Map<String, Double> MPQALexi = this.loadMPQA();
    	Map<String, Double> BingLiuLexi = this.loadBingLiu();
    	Map<String, Double> NRCLexi = this.loadNRC();
		
		int featurecount = 0;
		Map<String, Integer> nGramMap = new HashMap<String, Integer>();
		Map<String, Integer> CharNGramMap = new HashMap<String, Integer>();
		Map<String, Integer> posMap = new HashMap<String, Integer>();
		Map<String, Integer> clusterMap = new HashMap<String, Integer>();
		Map<String, Integer> emoticonMap = new HashMap<String, Integer>();
		ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
		
		//creating features
		for(Tweet tweet : this.tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.tokenizeAndTag(tagger, tweet);
			this.negate(tweet);
			
			//get n-grams and set n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 4);
			for (String nGram : nGramSet){
				if(!nGramMap.containsKey(nGram)){
					nGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("NGRAM_" + nGram));
				}
			}
			
			//get n-char-grams and set n-char-gram feature
			Set<String> CharNGramSet = this.getCharNGrams(tweet);
			for (String nGram : CharNGramSet){
				if(!CharNGramMap.containsKey(nGram)){
					CharNGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("CHARNGRAM_" + nGram));
				}
			}
			
			//get pos-tags and set pos-tag feature
			Map<String, Integer> posTags = this.getPosTags(tweet);
			for (Map.Entry<String, Integer> posTag : posTags.entrySet()){
				if(!posMap.containsKey(posTag.getKey())){
					posMap.put(posTag.getKey(), featurecount++);
					attributeList.add(new Attribute("POS_" + posTag.getKey()));
				}
			}
			
			//get cluster and set cluster feature
			Set<String> clusterSet = this.getClusters(tweet);
			for(String cluster : clusterSet){
				if(!clusterMap.containsKey(cluster)){
					clusterMap.put(cluster, featurecount++);
					attributeList.add(new Attribute("CLUSTER_" + cluster));
				}
			}
			
			//get emoticons and set emoticon feature
			Set<String> emoticonSet = this.getEmoticons(tweet);
			for(String emoticon : emoticonSet){
				if(!emoticonMap.containsKey(emoticon)){
					emoticonMap.put(emoticon, featurecount++);
					attributeList.add(new Attribute("EMO_" + emoticon));
				}
			}
		}
	
		//set allCaps feature
		Attribute allCaps = new Attribute("allCaps");
		attributeList.add(allCaps);
		featurecount++;
		
		//set hashtag feature
		Attribute hashtags = new Attribute("hashtags");
		attributeList.add(hashtags);
		featurecount++;
		
		//set punctuation features
		Attribute punctuationCount = new Attribute("punctuationCount");
		attributeList.add(punctuationCount);
		featurecount++;
		
		Attribute punctuationLast = new Attribute("punctuationLast");
		attributeList.add(punctuationLast);
		featurecount++;
		
		//set emoticon feature
		Attribute emoticonLast = new Attribute("emoticonLast");
		attributeList.add(emoticonLast);
		featurecount++;
		
		//set elongated words feature
		Attribute elongatedWords = new Attribute("elongatedWords");
		attributeList.add(elongatedWords);
		featurecount++;
		
	    //set negation feature
		Attribute negationCount = new Attribute("negationCount");
		attributeList.add(negationCount);
		featurecount++;
		
		//set lexica features	
		//senti140Uni
		Attribute senti140UniTotalCountPos = new Attribute("senti140UniTotalCountPos");
		attributeList.add(senti140UniTotalCountPos);
		featurecount++;
		
		Attribute senti140UniTotalScorePos = new Attribute("senti140UniTotalScorePos");
		attributeList.add(senti140UniTotalScorePos);
		featurecount++;
		
		Attribute senti140UniMaxScorePos = new Attribute("senti140UniMaxScorePos");
		attributeList.add(senti140UniMaxScorePos);
		featurecount++;
		
		Attribute senti140UniLastScorePos = new Attribute("senti140UniLastScorePos");
		attributeList.add(senti140UniLastScorePos);
		featurecount++;
		
		Attribute senti140UniTotalCountNeg = new Attribute("senti140UniTotalCountNeg");
		attributeList.add(senti140UniTotalCountNeg);
		featurecount++;
		
		Attribute senti140UniTotalScoreNeg = new Attribute("senti140UniTotalScoreNeg");
		attributeList.add(senti140UniTotalScoreNeg);
		featurecount++;
		
		Attribute senti140UniMaxScoreNeg = new Attribute("senti140UniMaxScoreNeg");
		attributeList.add(senti140UniMaxScoreNeg);
		featurecount++;
		
		Attribute senti140UniLastScoreNeg = new Attribute("senti140UniLastScoreNeg");
		attributeList.add(senti140UniLastScoreNeg);
		featurecount++;
		
		//hashtagUni
		Attribute hashtagUniTotalCountPos = new Attribute("hashtagUniTotalCountPos");
		attributeList.add(hashtagUniTotalCountPos);
		featurecount++;
		
		Attribute hashtagUniTotalScorePos = new Attribute("hashtagUniTotalScorePos");
		attributeList.add(hashtagUniTotalScorePos);
		featurecount++;
		
		Attribute hashtagUniMaxScorePos = new Attribute("hashtagUniMaxScorePos");
		attributeList.add(hashtagUniMaxScorePos);
		featurecount++;
		
		Attribute hashtagUniLastScorePos = new Attribute("hashtagUniLastScorePos");
		attributeList.add(hashtagUniLastScorePos);
		featurecount++;
		
		Attribute hashtagUniTotalCountNeg = new Attribute("hashtagUniTotalCountNeg");
		attributeList.add(hashtagUniTotalCountNeg);
		featurecount++;
		
		Attribute hashtagUniTotalScoreNeg = new Attribute("hashtagUniTotalScoreNeg");
		attributeList.add(hashtagUniTotalScoreNeg);
		featurecount++;
		
		Attribute hashtagUniMaxScoreNeg = new Attribute("hashtagUniMaxScoreNeg");
		attributeList.add(hashtagUniMaxScoreNeg);
		featurecount++;
		
		Attribute hashtagUniLastScoreNeg = new Attribute("hashtagUniLastScoreNeg");
		attributeList.add(hashtagUniLastScoreNeg);
		featurecount++;
		
		//senti140Bi
		Attribute senti140BiTotalCountPos = new Attribute("senti140BiTotalCountPos");
		attributeList.add(senti140BiTotalCountPos);
		featurecount++;
		
		Attribute senti140BiTotalScorePos = new Attribute("senti140BiTotalScorePos");
		attributeList.add(senti140BiTotalScorePos);
		featurecount++;
		
		Attribute senti140BiMaxScorePos = new Attribute("senti140BiMaxScorePos");
		attributeList.add(senti140BiMaxScorePos);
		featurecount++;
		
		Attribute senti140BiLastScorePos = new Attribute("senti140BiLastScorePos");
		attributeList.add(senti140BiLastScorePos);
		featurecount++;
		
		Attribute senti140BiTotalCountNeg = new Attribute("senti140BiTotalCountNeg");
		attributeList.add(senti140BiTotalCountNeg);
		featurecount++;
		
		Attribute senti140BiTotalScoreNeg = new Attribute("senti140BiTotalScoreNeg");
		attributeList.add(senti140BiTotalScoreNeg);
		featurecount++;
		
		Attribute senti140BiMaxScoreNeg = new Attribute("senti140BiMaxScoreNeg");
		attributeList.add(senti140BiMaxScoreNeg);
		featurecount++;
		
		Attribute senti140BiLastScoreNeg = new Attribute("senti140BiLastScoreNeg");
		attributeList.add(senti140BiLastScoreNeg);
		featurecount++;
		
		//hashtagBi
		Attribute hashtagBiTotalCountPos = new Attribute("hashtagBiTotalCountPos");
		attributeList.add(hashtagBiTotalCountPos);
		featurecount++;
		
		Attribute hashtagBiTotalScorePos = new Attribute("hashtagBiTotalScorePos");
		attributeList.add(hashtagBiTotalScorePos);
		featurecount++;
		
		Attribute hashtagBiMaxScorePos = new Attribute("hashtagBiMaxScorePos");
		attributeList.add(hashtagBiMaxScorePos);
		featurecount++;
		
		Attribute hashtagBiLastScorePos = new Attribute("hashtagBiLastScorePos");
		attributeList.add(hashtagBiLastScorePos);
		featurecount++;
		
		Attribute hashtagBiTotalCountNeg = new Attribute("hashtagBiTotalCountNeg");
		attributeList.add(hashtagBiTotalCountNeg);
		featurecount++;
		
		Attribute hashtagBiTotalScoreNeg = new Attribute("hashtagBiTotalScoreNeg");
		attributeList.add(hashtagBiTotalScoreNeg);
		featurecount++;
		
		Attribute hashtagBiMaxScoreNeg = new Attribute("hashtagBiMaxScoreNeg");
		attributeList.add(hashtagBiMaxScoreNeg);
		featurecount++;
		
		Attribute hashtagBiLastScoreNeg = new Attribute("hashtagBiLastScoreNeg");
		attributeList.add(hashtagBiLastScoreNeg);
		featurecount++;
		
		//MPQA
		Attribute MPQATotalCountPos = new Attribute("MPQATotalCountPos");
		attributeList.add(MPQATotalCountPos);
		featurecount++;
		
		Attribute MPQATotalScorePos = new Attribute("MPQATotalScorePos");
		attributeList.add(MPQATotalScorePos);
		featurecount++;
		
		Attribute MPQAMaxScorePos = new Attribute("MPQAMaxScorePos");
		attributeList.add(MPQAMaxScorePos);
		featurecount++;
		
		Attribute MPQALastScorePos = new Attribute("MPQALastScorePos");
		attributeList.add(MPQALastScorePos);
		featurecount++;
		
		Attribute MPQATotalCountNeg = new Attribute("MPQATotalCountNeg");
		attributeList.add(MPQATotalCountNeg);
		featurecount++;
		
		Attribute MPQATotalScoreNeg = new Attribute("MPQATotalScoreNeg");
		attributeList.add(MPQATotalScoreNeg);
		featurecount++;
		
		Attribute MPQAMaxScoreNeg = new Attribute("MPQAMaxScoreNeg");
		attributeList.add(MPQAMaxScoreNeg);
		featurecount++;
		
		Attribute MPQALastScoreNeg = new Attribute("MPQALastScoreNeg");
		attributeList.add(MPQALastScoreNeg);
		featurecount++;
		
		//BingLiu
		Attribute BingLiuTotalCountPos = new Attribute("BingLiuTotalCountPos");
		attributeList.add(BingLiuTotalCountPos);
		featurecount++;
		
		Attribute BingLiuTotalScorePos = new Attribute("BingLiuTotalScorePos");
		attributeList.add(BingLiuTotalScorePos);
		featurecount++;
		
		Attribute BingLiuMaxScorePos = new Attribute("BingLiuMaxScorePos");
		attributeList.add(BingLiuMaxScorePos);
		featurecount++;
		
		Attribute BingLiuLastScorePos = new Attribute("BingLiuLastScorePos");
		attributeList.add(BingLiuLastScorePos);
		featurecount++;
		
		Attribute BingLiuTotalCountNeg = new Attribute("BingLiuTotalCountNeg");
		attributeList.add(BingLiuTotalCountNeg);
		featurecount++;
		
		Attribute BingLiuTotalScoreNeg = new Attribute("BingLiuTotalScoreNeg");
		attributeList.add(BingLiuTotalScoreNeg);
		featurecount++;
		
		Attribute BingLiuMaxScoreNeg = new Attribute("BingLiuMaxScoreNeg");
		attributeList.add(BingLiuMaxScoreNeg);
		featurecount++;
		
		Attribute BingLiuLastScoreNeg = new Attribute("BingLiuLastScoreNeg");
		attributeList.add(BingLiuLastScoreNeg);
		featurecount++;
		
		//NRC
		Attribute NRCTotalCountPos = new Attribute("NRCTotalCountPos");
		attributeList.add(NRCTotalCountPos);
		featurecount++;
		
		Attribute NRCTotalScorePos = new Attribute("NRCTotalScorePos");
		attributeList.add(NRCTotalScorePos);
		featurecount++;
		
		Attribute NRCMaxScorePos = new Attribute("NRCMaxScorePos");
		attributeList.add(NRCMaxScorePos);
		featurecount++;
		
		Attribute NRCLastScorePos = new Attribute("NRCLastScorePos");
		attributeList.add(NRCLastScorePos);
		featurecount++;
		
		Attribute NRCTotalCountNeg = new Attribute("NRCTotalCountNeg");
		attributeList.add(NRCTotalCountNeg);
		featurecount++;
		
		Attribute NRCTotalScoreNeg = new Attribute("NRCTotalScoreNeg");
		attributeList.add(NRCTotalScoreNeg);
		featurecount++;
		
		Attribute NRCMaxScoreNeg = new Attribute("NRCMaxScoreNeg");
		attributeList.add(NRCMaxScoreNeg);
		featurecount++;
		
		Attribute NRCLastScoreNeg = new Attribute("NRCLastScoreNeg");
		attributeList.add(NRCLastScoreNeg);
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
			
			//char-n-gram feature
			Set<String> CharNGramSet = tweet.getCharNGramList();
			for (String nGram : CharNGramSet){
				Integer index = CharNGramMap.get(nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//pos-tag feature
			Map<String, Integer> posTags = tweet.getPosTagList();
			for (Map.Entry<String, Integer> posTag : posTags.entrySet()){
				Integer index = posMap.get(posTag.getKey());
				if(index != null){
					instance.setValue(index, posTag.getValue());
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
			
			//emoticon feature
			Set<String> emoticonSet = tweet.getEmoticonList();
			for(String emoticon : emoticonSet){
				Integer index = emoticonMap.get(emoticon);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			instance.setValue(allCaps, this.getAllCapsCount(tweet.getRawTweetString()));
			instance.setValue(hashtags, this.getHashtagCount(tweet.getTweetString()));
			instance.setValue(punctuationCount, this.getPunctuation(tweet.getTweetString()));
			if(this.isLastPunctuation(tweet.getTweetString())){
				instance.setValue(punctuationLast, 1);
			}
			else{
				instance.setValue(punctuationLast, 0);
			}
			if(tweet.isLastEmoticon()){
				instance.setValue(emoticonLast, 1);
			}
			else{
				instance.setValue(emoticonLast, 0);
			}
			instance.setValue(elongatedWords, this.getElongatedCount(tweet.getTweetString()));
			instance.setValue(negationCount, tweet.getNegationCount());
			
			//lexica feature
			List<Double> senti140UniPos = this.getLexiScores(senti140UniLexi, tweet.getWordList(), false);
			instance.setValue(senti140UniTotalCountPos, senti140UniPos.get(0));
			instance.setValue(senti140UniTotalScorePos, senti140UniPos.get(1));
			instance.setValue(senti140UniMaxScorePos, senti140UniPos.get(2));
			instance.setValue(senti140UniLastScorePos, senti140UniPos.get(3));
			List<Double> hashtagUniPos = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), false);
			instance.setValue(hashtagUniTotalCountPos, hashtagUniPos.get(0));
			instance.setValue(hashtagUniTotalScorePos, hashtagUniPos.get(1));
			instance.setValue(hashtagUniMaxScorePos, hashtagUniPos.get(2));
			instance.setValue(hashtagUniLastScorePos, hashtagUniPos.get(3));
			List<Double> senti140UniNeg = this.getLexiScores(senti140UniLexi, tweet.getWordList(), true);
			instance.setValue(senti140UniTotalCountNeg, senti140UniNeg.get(0));
			instance.setValue(senti140UniTotalScoreNeg, senti140UniNeg.get(1));
			instance.setValue(senti140UniMaxScoreNeg, senti140UniNeg.get(2));
			instance.setValue(senti140UniLastScoreNeg, senti140UniNeg.get(3));
			List<Double> hashtagUniNeg = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), true);
			instance.setValue(hashtagUniTotalCountNeg, hashtagUniNeg.get(0));
			instance.setValue(hashtagUniTotalScoreNeg, hashtagUniNeg.get(1));
			instance.setValue(hashtagUniMaxScoreNeg, hashtagUniNeg.get(2));
			instance.setValue(hashtagUniLastScoreNeg, hashtagUniNeg.get(3));
			
			Set<String> biGramSet = this.getNGrams(tweet, 2, 2);
			List<Double> senti140BiPos = this.getLexiScoresBi(senti140BiLexi, biGramSet, false);
			instance.setValue(senti140BiTotalCountPos, senti140BiPos.get(0));
			instance.setValue(senti140BiTotalScorePos, senti140BiPos.get(1));
			instance.setValue(senti140BiMaxScorePos, senti140BiPos.get(2));
			instance.setValue(senti140BiLastScorePos, senti140BiPos.get(3));
			List<Double> hashtagBiPos = this.getLexiScoresBi(hashtagBiLexi, biGramSet, false);
			instance.setValue(hashtagBiTotalCountPos, hashtagBiPos.get(0));
			instance.setValue(hashtagBiTotalScorePos, hashtagBiPos.get(1));
			instance.setValue(hashtagBiMaxScorePos, hashtagBiPos.get(2));
			instance.setValue(hashtagBiLastScorePos, hashtagBiPos.get(3));
			List<Double> senti140BiNeg = this.getLexiScoresBi(senti140BiLexi, biGramSet, true);
			instance.setValue(senti140BiTotalCountNeg, senti140BiNeg.get(0));
			instance.setValue(senti140BiTotalScoreNeg, senti140BiNeg.get(1));
			instance.setValue(senti140BiMaxScoreNeg, senti140BiNeg.get(2));
			instance.setValue(senti140BiLastScoreNeg, senti140BiNeg.get(3));
			List<Double> hashtagBiNeg = this.getLexiScoresBi(hashtagBiLexi, biGramSet, true);
			instance.setValue(hashtagBiTotalCountNeg, hashtagBiNeg.get(0));
			instance.setValue(hashtagBiTotalScoreNeg, hashtagBiNeg.get(1));
			instance.setValue(hashtagBiMaxScoreNeg, hashtagBiNeg.get(2));
			instance.setValue(hashtagBiLastScoreNeg, hashtagBiNeg.get(3));
			
			List<Double> MPQAPos = this.getLexiScores(MPQALexi, tweet.getWordList(), false);
			instance.setValue(MPQATotalCountPos, MPQAPos.get(0));
			instance.setValue(MPQATotalScorePos, MPQAPos.get(1));
			instance.setValue(MPQAMaxScorePos, MPQAPos.get(2));
			instance.setValue(MPQALastScorePos, MPQAPos.get(3));
			List<Double> MPQANeg = this.getLexiScores(MPQALexi, tweet.getWordList(), true);
			instance.setValue(MPQATotalCountNeg, MPQANeg.get(0));
			instance.setValue(MPQATotalScoreNeg, MPQANeg.get(1));
			instance.setValue(MPQAMaxScoreNeg, MPQANeg.get(2));
			instance.setValue(MPQALastScoreNeg, MPQANeg.get(3));
			
			List<Double> BingLiuPos = this.getLexiScores(BingLiuLexi, tweet.getWordList(), false);
			instance.setValue(BingLiuTotalCountPos, BingLiuPos.get(0));
			instance.setValue(BingLiuTotalScorePos, BingLiuPos.get(1));
			instance.setValue(BingLiuMaxScorePos, BingLiuPos.get(2));
			instance.setValue(BingLiuLastScorePos, BingLiuPos.get(3));
			List<Double> BingLiuNeg = this.getLexiScores(BingLiuLexi, tweet.getWordList(), true);
			instance.setValue(BingLiuTotalCountNeg, BingLiuNeg.get(0));
			instance.setValue(BingLiuTotalScoreNeg, BingLiuNeg.get(1));
			instance.setValue(BingLiuMaxScoreNeg, BingLiuNeg.get(2));
			instance.setValue(BingLiuLastScoreNeg, BingLiuNeg.get(3));
			
			List<Double> NRCPos = this.getLexiScores(NRCLexi, tweet.getWordList(), false);
			instance.setValue(NRCTotalCountPos, NRCPos.get(0));
			instance.setValue(NRCTotalScorePos, NRCPos.get(1));
			instance.setValue(NRCMaxScorePos, NRCPos.get(2));
			instance.setValue(NRCLastScorePos, NRCPos.get(3));
			List<Double> NRCNeg = this.getLexiScores(NRCLexi, tweet.getWordList(), true);
			instance.setValue(NRCTotalCountNeg, NRCNeg.get(0));
			instance.setValue(NRCTotalScoreNeg, NRCNeg.get(1));
			instance.setValue(NRCMaxScoreNeg, NRCNeg.get(2));
			instance.setValue(NRCLastScoreNeg, NRCNeg.get(3));
			
			//set class attribute
			instance.setValue(classAttribute, tweet.getSentiment());

			trainingSet.add(instance);
		}
		
		//save features and training instances in .arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingSet);
		saver.setFile(new File("resources/arff/Trained-Features-" + "NRC" + savename + ".arff"));
		saver.writeBatch();
		System.out.println("Trained-Features-" + "NRC" + savename + " saved");
	}
	
    /**
     * Creates all features and instances for the testdata and classifies the Tweet
     * 
     * @param nameOfTrain optional filename of the arff file
     * @return returns all results in a map
     * @throws Exception
     */ 	
	public Map<String,ClassificationResult> test(String nameOfTrain) throws Exception{
		System.out.println("Starting NRC Test");
		System.out.println("Tweets: " +  this.tweetList.size());
		String trainname = "";
		if(!nameOfTrain.equals("")){
			trainname = nameOfTrain;
		}
		else{
			trainname = "Trained-Features-NRC";
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
        classifier.setCost(0.5);
        
        //train classifier with instances
        classifier.buildClassifier(train);
        
        //delete train instances, to use same features with test instances
		train.delete();
		
		//load pos-tagger
    	Tagger tagger = new Tagger();
    	tagger.loadModel("resources/tagger/model.20120919.txt");
    	
    	//load sentiment lexica
    	Map<String, Double> senti140UniLexi = this.loadLexicon("sentiment140/unigrams-pmilexicon");
    	Map<String, Double> hashtagUniLexi = this.loadLexicon("hashtag/unigrams-pmilexicon");
    	Map<String, Double> senti140BiLexi = this.loadLexicon("sentiment140/bigrams-pmilexicon");
    	Map<String, Double> hashtagBiLexi = this.loadLexicon("hashtag/bigrams-pmilexicon");
    	Map<String, Double> MPQALexi = this.loadMPQA();
    	Map<String, Double> BingLiuLexi = this.loadBingLiu();
    	Map<String, Double> NRCLexi = this.loadNRC();
		
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		for (int i = 0; i < train.numAttributes(); i++){
			featureMap.put(train.attribute(i).name(), train.attribute(i).index());
		}
	    
		Map<String,ClassificationResult> resultMap = new HashMap<String,ClassificationResult>();
		for(Tweet tweet : this.tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.tokenizeAndTag(tagger, tweet);
			this.negate(tweet);
			SparseInstance instance = new SparseInstance(0);
			
			//creating test instances with features
            //n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 4);
			for (String nGram : nGramSet){
				Integer index = featureMap.get("NGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//char-n-gram feature
			Set<String> CharNGramSet = this.getCharNGrams(tweet);
			for (String nGram : CharNGramSet){
				Integer index = featureMap.get("CHARNGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//pos-tag feature
			Map<String, Integer> posTags = this.getPosTags(tweet);
			for (Map.Entry<String, Integer> posTag : posTags.entrySet()){
				Integer index = featureMap.get("POS_" +posTag.getKey());
				if(index != null){
					instance.setValue(index, posTag.getValue());
				}
			}
			
			//cluster feature
			Set<String> clusterSet = this.getClusters(tweet);
			for(String cluster : clusterSet){
				Integer index = featureMap.get("CLUSTER_" + cluster);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//emoticon feature
			Set<String> emoticonSet = this.getEmoticons(tweet);
			for(String emoticon : emoticonSet){
				Integer index = featureMap.get("EMO_" + emoticon);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			instance.setValue(featureMap.get("allCaps"), this.getAllCapsCount(tweet.getRawTweetString()));
			instance.setValue(featureMap.get("hashtags"), this.getHashtagCount(tweet.getTweetString()));
			instance.setValue(featureMap.get("punctuationCount"), this.getPunctuation(tweet.getTweetString()));
			if(this.isLastPunctuation(tweet.getTweetString())){
				instance.setValue(featureMap.get("punctuationLast"), 1);
			}
			else{
				instance.setValue(featureMap.get("punctuationLast"), 0);
			}
			if(tweet.isLastEmoticon()){
				instance.setValue(featureMap.get("emoticonLast"), 1);
			}
			else{
				instance.setValue(featureMap.get("emoticonLast"), 0);
			}
			instance.setValue(featureMap.get("elongatedWords"), this.getElongatedCount(tweet.getTweetString()));
			instance.setValue(featureMap.get("negationCount"), tweet.getNegationCount());
			
			//lexica features
			List<Double> senti140UniPos = this.getLexiScores(senti140UniLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("senti140UniTotalCountPos"), senti140UniPos.get(0));
			instance.setValue(featureMap.get("senti140UniTotalScorePos"), senti140UniPos.get(1));
			instance.setValue(featureMap.get("senti140UniMaxScorePos"), senti140UniPos.get(2));
			instance.setValue(featureMap.get("senti140UniLastScorePos"), senti140UniPos.get(3));
			List<Double> hashtagUniPos = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("hashtagUniTotalCountPos"), hashtagUniPos.get(0));
			instance.setValue(featureMap.get("hashtagUniTotalScorePos"), hashtagUniPos.get(1));
			instance.setValue(featureMap.get("hashtagUniMaxScorePos"), hashtagUniPos.get(2));
			instance.setValue(featureMap.get("hashtagUniLastScorePos"), hashtagUniPos.get(3));
			List<Double> senti140UniNeg = this.getLexiScores(senti140UniLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("senti140UniTotalCountNeg"), senti140UniNeg.get(0));
			instance.setValue(featureMap.get("senti140UniTotalScoreNeg"), senti140UniNeg.get(1));
			instance.setValue(featureMap.get("senti140UniMaxScoreNeg"), senti140UniNeg.get(2));
			instance.setValue(featureMap.get("senti140UniLastScoreNeg"), senti140UniNeg.get(3));
			List<Double> hashtagUniNeg = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("hashtagUniTotalCountNeg"), hashtagUniNeg.get(0));
			instance.setValue(featureMap.get("hashtagUniTotalScoreNeg"), hashtagUniNeg.get(1));
			instance.setValue(featureMap.get("hashtagUniMaxScoreNeg"), hashtagUniNeg.get(2));
			instance.setValue(featureMap.get("hashtagUniLastScoreNeg"), hashtagUniNeg.get(3));
			
			Set<String> biGramSet = this.getNGrams(tweet, 2, 2);
			List<Double> senti140BiPos = this.getLexiScoresBi(senti140BiLexi, biGramSet, false);
			instance.setValue(featureMap.get("senti140BiTotalCountPos"), senti140BiPos.get(0));
			instance.setValue(featureMap.get("senti140BiTotalScorePos"), senti140BiPos.get(1));
			instance.setValue(featureMap.get("senti140BiMaxScorePos"), senti140BiPos.get(2));
			instance.setValue(featureMap.get("senti140BiLastScorePos"), senti140BiPos.get(3));
			List<Double> hashtagBiPos = this.getLexiScoresBi(hashtagBiLexi, biGramSet, false);
			instance.setValue(featureMap.get("hashtagBiTotalCountPos"), hashtagBiPos.get(0));
			instance.setValue(featureMap.get("hashtagBiTotalScorePos"), hashtagBiPos.get(1));
			instance.setValue(featureMap.get("hashtagBiMaxScorePos"), hashtagBiPos.get(2));
			instance.setValue(featureMap.get("hashtagBiLastScorePos"), hashtagBiPos.get(3));
			List<Double> senti140BiNeg = this.getLexiScoresBi(senti140BiLexi, biGramSet, true);
			instance.setValue(featureMap.get("senti140BiTotalCountNeg"), senti140BiNeg.get(0));
			instance.setValue(featureMap.get("senti140BiTotalScoreNeg"), senti140BiNeg.get(1));
			instance.setValue(featureMap.get("senti140BiMaxScoreNeg"), senti140BiNeg.get(2));
			instance.setValue(featureMap.get("senti140BiLastScoreNeg"), senti140BiNeg.get(3));
			List<Double> hashtagBiNeg = this.getLexiScoresBi(hashtagBiLexi, biGramSet, true);
			instance.setValue(featureMap.get("hashtagBiTotalCountNeg"), hashtagBiNeg.get(0));
			instance.setValue(featureMap.get("hashtagBiTotalScoreNeg"), hashtagBiNeg.get(1));
			instance.setValue(featureMap.get("hashtagBiMaxScoreNeg"), hashtagBiNeg.get(2));
			instance.setValue(featureMap.get("hashtagBiLastScoreNeg"), hashtagBiNeg.get(3));
			
			List<Double> MPQAPos = this.getLexiScores(MPQALexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("MPQATotalCountPos"), MPQAPos.get(0));
			instance.setValue(featureMap.get("MPQATotalScorePos"), MPQAPos.get(1));
			instance.setValue(featureMap.get("MPQAMaxScorePos"), MPQAPos.get(2));
			instance.setValue(featureMap.get("MPQALastScorePos"), MPQAPos.get(3));
			List<Double> MPQANeg = this.getLexiScores(MPQALexi,  tweet.getWordList(), true);
			instance.setValue(featureMap.get("MPQATotalCountNeg"), MPQANeg.get(0));
			instance.setValue(featureMap.get("MPQATotalScoreNeg"), MPQANeg.get(1));
			instance.setValue(featureMap.get("MPQAMaxScoreNeg"), MPQANeg.get(2));
			instance.setValue(featureMap.get("MPQALastScoreNeg"), MPQANeg.get(3));
			
			List<Double> BingLiuPos = this.getLexiScores(BingLiuLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("BingLiuTotalCountPos"), BingLiuPos.get(0));
			instance.setValue(featureMap.get("BingLiuTotalScorePos"), BingLiuPos.get(1));
			instance.setValue(featureMap.get("BingLiuMaxScorePos"), BingLiuPos.get(2));
			instance.setValue(featureMap.get("BingLiuLastScorePos"), BingLiuPos.get(3));
			List<Double> BingLiuNeg = this.getLexiScores(BingLiuLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("BingLiuTotalCountNeg"), BingLiuNeg.get(0));
			instance.setValue(featureMap.get("BingLiuTotalScoreNeg"), BingLiuNeg.get(1));
			instance.setValue(featureMap.get("BingLiuMaxScoreNeg"), BingLiuNeg.get(2));
			instance.setValue(featureMap.get("BingLiuLastScoreNeg"), BingLiuNeg.get(3));
			
			List<Double> NRCPos = this.getLexiScores(NRCLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("NRCTotalCountPos"), NRCPos.get(0));
			instance.setValue(featureMap.get("NRCTotalScorePos"), NRCPos.get(1));
			instance.setValue(featureMap.get("NRCMaxScorePos"), NRCPos.get(2));
			instance.setValue(featureMap.get("NRCLastScorePos"), NRCPos.get(3));
			List<Double> NRCNeg = this.getLexiScores(NRCLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("NRCTotalCountNeg"), NRCNeg.get(0));
			instance.setValue(featureMap.get("NRCTotalScoreNeg"), NRCNeg.get(1));
			instance.setValue(featureMap.get("NRCMaxScoreNeg"), NRCNeg.get(2));
			instance.setValue(featureMap.get("NRCLastScoreNeg"), NRCNeg.get(3));
			
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
    }
    
	/**
	 * Gets the PoS-Tags for the pos tag feature
	 * 
	 * @param tweet the Tweet to analyze
	 * @return returns a list with pos tags that occur in the Tweet
	 */
    private Map<String, Integer> getPosTags(Tweet tweet){
    	Map<String,Integer> tagMap = new HashMap<String, Integer>();
    	for (TaggedToken token : tweet.getWordList()){
        	Integer val = tagMap.get(token.tag);
        	if (val != null){
        		tagMap.put(token.tag, ++val);
        	}
        	else{
        		tagMap.put(token.tag, 1);
        	}
    	}
    	tweet.setPosTags(tagMap);
	    return tagMap;
    }
    
    /**
     * Determine the words which all in caps
     * 
     * @param tweetString the Tweetstring to analyze
     * @return returns the number of all caps words
     */
    private int getAllCapsCount(String tweetString){
    	int wordsInCaps = 0;
    	for (String word: tweetString.split("[\\p{P} \\t\\n\\r]")){
    		if(word == word.toUpperCase() && word.length() > 1) wordsInCaps++;
    	}
    	return wordsInCaps;
    }
    
    
    /**
     * Determine how many hashtags are in the Tweet
     * 
     * @param tweetString the Tweetstring to analyze
     * @return returns the number of hashtags in the Tweet
     */
    private int getHashtagCount(String tweetString){
        String tempString = tweetString;
        return tempString.length() - tempString.replace("#", "").length();
    }
    
    /**
     * Determine how many punctuations are in the Tweet
     *     
     * @param tweetString the Tweetstring to analyze
     * @return returns the number of punctuations in the Tweet
     */
    private int getPunctuation(String tweetString) {
		int punctuations = 0;
		Matcher m = Pattern.compile("[!,?]{2,}").matcher(tweetString);
		while (m.find()){
			punctuations++;
		}
		return punctuations;
	}

    /**
     * Determine if the Tweet ends with an ? or !
     * 
     * @param tweetString the Tweetstring to analyze
     * @return returns true if the Tweet ends with ? or ! and false if not
     */
	private boolean isLastPunctuation(String tweetString) {
		if (tweetString.endsWith("?") || tweetString.endsWith("!")){
			return true;
		}
		return false;
	}
	
	/**
	 * Determine how many elongatedwords are in the Tweet
	 * 
	 * @param tweetString the Tweetstring to analyze
	 * @return returns the number of elongatedwords in the Tweet
	 */
	private int getElongatedCount(String tweetString){
    	int elongatedWords = 0;
    	for (String word: tweetString.split("[\\p{P} \\t\\n\\r]")){
    		Matcher m = Pattern.compile("(.)\\1{2,}").matcher(word);    		
    		if(m.find()) elongatedWords++;
    	}
    	return elongatedWords;
    }
}
