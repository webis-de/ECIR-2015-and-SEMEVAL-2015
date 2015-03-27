import java.util.List;
import java.util.Map;
import java.util.Set;

import cmu.arktweetnlp.Tagger.TaggedToken;

/**
 * Represents one Tweet and saves some features and the preprocessed versions of the Tweet
 */
public class Tweet {
	private String rawTweet;
	private String tweetID;
    private String tweetString;
    private String sentiment;
    private List<TaggedToken> wordList;
    private List<TaggedToken> wordListRaw;
    private List<TaggedToken> wordListCollapsed;
    private Map<String, String> wordListStanford;
    private boolean lastEmoticon = false;
    private int negationCount = 0;
	private Set<String> nGramList;
	private Set<String> charNGramList;
	private Map<String, Integer> posTagList;
	private Set<String> clusterList;
	private Set<String> emoticonList;
	private Set<String> stemList;
    
    public Tweet(String tweet, String senti, String tID){
    	rawTweet = tweet;
        sentiment = senti;
    	tweetID = tID;
    }
    
    public String getRawTweetString() {
		return this.rawTweet;
	}
    
    public String getTweetID() {
        return this.tweetID;
    }
    
    public String getTweetString() {
		return this.tweetString;
	}
    
    public void setTweetString(String tstring) {
		this.tweetString = tstring;
	}
    
    public String getSentiment() {
        return this.sentiment;
    }
    
    public void setWordList(List<TaggedToken> wList){
    	this.wordList = wList;
    }
            
    public List<TaggedToken> getWordList() {
	  	return this.wordList;
  	}
    
    public List<TaggedToken> getRawWordList() {
	  	return this.wordListRaw;
  	}
    
	public void setRawWordList(List<TaggedToken> wListRaw) {
		this.wordListRaw = wListRaw;
	}
	
    public List<TaggedToken> getCollapsedWordList() {
	  	return this.wordListCollapsed;
  	}
    
	public void setCollapseList(List<TaggedToken> wListCol) {
		this.wordListCollapsed = wListCol;
	}

	public void setStanfordWordList(Map<String, String> sList) {
		this.wordListStanford = sList;
		
	}
	public Map<String, String> getStanfordWordList() {
		return this.wordListStanford;
	}
	
	public void setLastEmoticon(boolean b) {
		this.lastEmoticon = b;
	}
	
	public boolean isLastEmoticon(){
		return this.lastEmoticon;
	}
	
	public int getNegationCount() {
		return this.negationCount;
	}

	public void setNegationCount(int nCount) {
		this.negationCount = nCount;
	}
	
	public Set<String> getnGramList() {
		return this.nGramList;
	}
	
	public void setNGrams(Set<String> nGList) {
		this.nGramList = nGList;
	}
	
	public Set<String> getCharNGramList() {
		return this.charNGramList;
	}
    
	public void setCharNGramList(Set<String> nGramList) {
		this.charNGramList = nGramList;
	}
	
	public Map<String, Integer> getPosTagList() {
		return this.posTagList;
	}
	
	public void setPosTags(Map<String, Integer> tagMap) {
		this.posTagList = tagMap;
		
	}

	public Set<String> getClusterList() {
		return this.clusterList;
	}
	
	public void setClusters(Set<String> cList) {
		this.clusterList = cList;
		
	}

	public Set<String> getEmoticonList() {
		return this.emoticonList;
	}
	
	public void setEmoticons(Set<String> emoticons) {
		this.emoticonList = emoticons;
	}
	
	public Set<String> getStemList() {
		return this.stemList;
	}

	public void setStemList(Set<String> sList) {
		this.stemList = sList;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((tweetString == null) ? 0 : tweetString.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Tweet other = (Tweet) obj;
		if (this.rawTweet == null) {
			if (other.rawTweet != null)
				return false;
		} else if (!this.tweetID.equals(other.tweetID))
			return false;
		return true;
	}
}
