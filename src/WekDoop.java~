import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.j48.*;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.Console;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.List; 

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.TaskID;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.InputSplit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;

import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.InputSplit;

/**
 * This is the main class that runs the whole program. It is responsible
 * for getting input from the user, setting up the mapper and reducer,
 * organizing the weka input, etc.
 */
public class WekDoop {
	public static class WekaInputFormat extends TextInputFormat {
		/**
		 * Takes a JobContext and returns a list of data split into pieces
		 * Basically this is a way of handling large data sets. This method allows
		 * us to split a large data set into smaller chunks to pass across worker nodes
		 * (or in our case, just to make life a little easier and pass the chunks to a single
		 * node so that it is not overwhelmed by one large data set)
		 * 
		 * @see org.apache.hadoop.mapreduce.lib.input.FileInputFormat#getSplits
		 * (org.apache.hadoop.mapreduce.JobContext)
		 */
	    public List<InputSplit> getSplits(JobContext job) throws IOException {
	        long minSize = Math.max(getFormatMinSplitSize(), getMinSplitSize(job));
	        long maxSize = getMaxSplitSize(job);
	        
	        List<InputSplit> splits = new ArrayList<InputSplit>();
	        for (FileStatus file: listStatus(job)) {
	            Path path = file.getPath();
	            FileSystem fs = path.getFileSystem(job.getConfiguration());

                //number of bytes in this file
                long length = file.getLen();
	            BlockLocation[] blkLocations = fs.getFileBlockLocations(file, 0, length);

	            // make sure this is actually a valid file
	            if(length != 0) {
	            	// set the number of splits to make. NOTE: the value can be changed to anything
	                int count = job.getConfiguration().getInt("Run-num.splits", 1);
	                for(int t = 0; t < count; t++) {
	                	//split the file and add each chunk to the list
	                    splits.add(new FileSplit(path, 0, length, blkLocations[0].getHosts())); 
                    }
	            }
                else {
	                // Create empty array for zero length files
	                splits.add(new FileSplit(path, 0, length, new String[0]));
	            }
	        }
	        return splits;
	    }
	}	
	
	/**
	 * This class is a mapper for the weka classifiers
	 * It is given a chunk of data and it sets up a classifier to run on that data.
	 * There is a lot of other handling that occurs in the method as well.
	 * 
	 * @author Trevor Hodde
	 */
	public static class WekaMap extends Mapper<Object, Text, Text, AggregateableEvaluation> {
	    private Instances randData = null;
	    private Classifier cls = null;

	    private AggregateableEvaluation eval = null;
	    private Classifier clsCopy = null;
	    
	    // Run 10 mappers
	    private String numMaps = "10";
	    
	    // TODO: Make sure this is not hard-coded -- preferably a command line arg
	    // Set the classifier
	    private String classname = "weka.classifiers.bayes.NaiveBayes";
	    private int seed = 20;
	    
	    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	        String line = value.toString();
			System.out.println("CURRENT LINE: " + line);

			//line = "/home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/spambase_processed.arff";
	      
            Configuration conf = new Configuration();
            FileSystem fileSystem = FileSystem.get(conf);

			Path path = new Path("/home/ubuntu/Workspace/hadoop-1.1.0/hadoop-data/very_small_spam.arff");

            // Make sure the file exists...
            if (!fileSystem.exists(path)) {
                System.out.println("File does not exists");
                return;
            }
	      
    	    JobID test = context.getJobID();
	        TaskAttemptID tid = context.getTaskAttemptID();
	      
	        // Set up the weka configuration
	        Configuration wekaConfig = context.getConfiguration();
	        numMaps = wekaConfig.get("Run-num.splits");
	        classname = wekaConfig.get("Run.classify");
	      
	        String[] splitter = tid.toString().split("_");
	        String jobNumber = "";
	        int n = 0;
	      
	        if (splitter[4].length() > 0) {
	    	    jobNumber = splitter[4].substring(splitter[4].length() - 1);
	    	    n = Integer.parseInt(jobNumber);
	        }
	      
	        FileSystem fs = FileSystem.get(context.getConfiguration());

			System.out.println("PATH: " + path);

	        // Read in the data set
	        context.setStatus("Reading in the arff file...");
			readArff(fs, path.toString());
	        context.setStatus("Done reading arff! Initializing aggregateable eval...");

            try {
			    eval = new AggregateableEvaluation(randData);
	        }
	        catch (Exception e1) {
			    e1.printStackTrace();
	        }
	      
            // Split the data into two sets: Training set and a testing set
            // this will allow us to use a little bit of data to train the classifier
            // before running the classifier on the rest of the dataset
	        Instances trainInstance = randData.trainCV(Integer.parseInt(numMaps), n);
	        Instances testInstance = randData.testCV(Integer.parseInt(numMaps), n);
	      
	        // Set parameters to be passed to the classifiers
	        String[] opts = new String[3];
	        if (classname.equals("weka.classifiers.lazy.IBk")) {
		        opts[0] = "";
		        opts[1] = "-K";
		        opts[2] = "1";
	        }
	        else if (classname.equals("weka.classifiers.trees.J48")) {
	    	    opts[0] = "";
		        opts[1] = "-C";
		        opts[2] = "0.25";
	        }
	        else if (classname.equals("weka.classifiers.bayes.NaiveBayes")) {
	    	    opts[0] = "";
		        opts[1] = "";
		        opts[2] = "";
	        }
            else {
                opts[0] = "";
                opts[1] = "";
                opts[2] = "";
            }
	      
	        // Start setting up the classifier and its various options
	        try {
			  cls = (Classifier) Utils.forName(Classifier.class, classname, opts);
	        }
	        catch (Exception e) {
			    e.printStackTrace();
	        }
	      
	        // These are all used for timing different processes
	        long beforeAbstract = 0;
	        long beforeBuildClass = 0;
	        long afterBuildClass = 0;
	        long beforeEvalClass = 0;
	        long afterEvalClass = 0;
	      
	        try {
	        	// Create the classifier and record how long it takes to set up 
	    	    context.setStatus("Creating the classifier...");
	    	    System.out.println(new Timestamp(System.currentTimeMillis()));
	    	    beforeAbstract = System.currentTimeMillis();
	    	    clsCopy = AbstractClassifier.makeCopy(cls);
	    	    beforeBuildClass = System.currentTimeMillis();
	    	    System.out.println(new Timestamp(System.currentTimeMillis()));
		      
	    	    // Train the classifier on the training set and record how long this takes
	    	    context.setStatus("Training the classifier...");
	    	    clsCopy.buildClassifier(trainInstance);
		        afterBuildClass = System.currentTimeMillis();
		        System.out.println(new Timestamp(System.currentTimeMillis()));
		        beforeEvalClass = System.currentTimeMillis();
		      
		        // Run the classifer on the rest of the data set and record its duration as well
		        context.setStatus("Evaluating the model...");
		        eval.evaluateModel(clsCopy, testInstance);
		        afterEvalClass = System.currentTimeMillis();
		        System.out.println(new Timestamp(System.currentTimeMillis()));

		        // We are done this iteration!
		        context.setStatus("Complete");
	        }
	        catch (Exception e) {
	    	    System.out.println("Debugging strarts here!");
	    	    e.printStackTrace();
	        }
	      
	        // calculate the total times for each section
	        long abstractTime = beforeBuildClass - beforeAbstract;
	        long buildTime = afterBuildClass - beforeBuildClass;
	        long evalTime = afterEvalClass - beforeEvalClass;
	      
	        // Print out the times
	        System.out.println("The value of creation time: " + abstractTime);
	        System.out.println("The value of Build time: " + buildTime);
	        System.out.println("The value of Eval time: " + evalTime);
	          
	        context.write(new Text(line), eval);
	      }
	    
	    /**
	     * This can be used to write out the results on HDFS, but it is not essential
	     * to the success of this project. If time allows, we can implement it.
	     */
	      public void writeResult()	{    }
	    
	    
	      /**
	       * This method reads in the arff file that is provided to the program.
	       * Nothing really special about the way the data is handled.
	       * 
	       * @param fs
	       * @param filePath
	       * @throws IOException
	       * @throws InterruptedException
	       */
	      public void readArff(FileSystem fs, String filePath) throws IOException, InterruptedException {
	    	  BufferedReader reader;
	    	  DataInputStream d;
	    	  ArffReader arff;
	    	  Instance inst;
	    	  Instances data;
	
	    	  try {
	    		  // Read in the data using a ton of wrappers
	    		  d = new DataInputStream(fs.open(new Path(filePath)));
			      reader = new BufferedReader(new InputStreamReader(d));
			      arff = new ArffReader(reader, 100000);
			      data = arff.getStructure();
			      data.setClassIndex(data.numAttributes() - 1);
			    
			      // Add each line to the input stream
			      while ((inst = arff.readInstance(data)) != null) {
			          data.add(inst);
			      }
			    		    
			      reader.close();
			    
			      Random rand = new Random(seed);
			      randData = new Instances(data);
			      randData.randomize(rand);

			      // This is how weka handles the sampling of the data
			      // the stratify method splits up the data to cross validate it
                  if (randData.classAttribute().isNominal()) {
			          randData.stratify(Integer.parseInt(numMaps));
                  }
	    	  }
	    	  catch (IOException e) {
	    		  e.printStackTrace();
	    	  }
	    }
	}
	
	/**
	 * This class is a reducer for the output from the weka classifiers
	 * It is given bunch of cross-validated data chunks from the mappers and its
	 * job is to aggregate the data into one solution.
	 *
	 * @author James Forkey
	 */
	public static class WekaReducer extends Reducer<Text, AggregateableEvaluation, Text, IntWritable> {
		private Text result = new Text();
		private Evaluation evalAll = null;
		private IntWritable test = new IntWritable();
		
		private AggregateableEvaluation aggEval;
			
		/**
		 * The reducer method takes all the stratified, cross-validated
		 * values from the mappers in a list and uses an aggregatable evaluation to consolidate
		 * them.
		 */
		public void reduce(Text key, Iterable<AggregateableEvaluation> values, Context context) throws IOException, InterruptedException {		
			int sum = 0;
			
			// record how long it takes to run the aggregation
			System.out.println(new Timestamp(System.currentTimeMillis()));
			long beforeReduceTime = System.currentTimeMillis();
			
			// loop through each of the values and "aggregate"
			// which basically means to consolidate the values
			for (AggregateableEvaluation val : values) {
				System.out.println("IN THE REDUCER!");

				// The first time through, give aggEval a value
				if (sum == 0) {
					try {
						aggEval = val;
					}
                    catch (Exception e) {
						e.printStackTrace();
					}
				}
				else {
					// combine the values
					aggEval.aggregate(val);
				}
				
				try {
					// This is what is taken from the mapper to be aggregated
					System.out.println("This is the map result");
					System.out.println(aggEval.toMatrixString());
				}
                catch (Exception e) {
					e.printStackTrace();
				}						
				
				sum += 1;
			}
			
			// Here is where the typical weka matrix output is generated
			try {
				System.out.println("This is reduce matrix");
				System.out.println(aggEval.toMatrixString());
			}
            catch (Exception e) {
				e.printStackTrace();
			}
			
			// calculate the duration of the aggregation
			context.write(key, new IntWritable(sum));
			long afterReduceTime = System.currentTimeMillis();
			long reduceTime = afterReduceTime - beforeReduceTime;
			
			// display the output
			System.out.println("The value of reduce time is: " + reduceTime);
			System.out.println(new Timestamp(System.currentTimeMillis()));
		}
	}
	
	/**
	 * The main method of this program. 
	 * Precondition: arff file is uploaded into HDFS and the correct
	 * number of parameters were passed into the JAR file when it was run
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
	    
		// Make sure we have the correct number of arguments passed into the program
		if (args.length != 4) {
	      System.err.println("Usage: WekDoop <# of splits> <classifier> <input file> <output file>");
	      System.exit(1);
	    }
		
		// configure the job using the command line args
	    conf.setInt("Run-num.splits", Integer.parseInt(args[0]));
	    conf.setStrings("Run.classify", args[1]);
	    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization," + "org.apache.hadoop.io.serializer.WritableSerialization");
	    
	    // Configure the jobs main class, mapper and reducer
		// TODO: Make the Job name print the name of the currently running classifier
	    Job job = new Job(conf, "WekDoop");
	    job.setJarByClass(WekDoop.class);
	    job.setMapperClass(WekaMap.class);
	    job.setReducerClass(WekaReducer.class);
	    
	    // Start with 1
	    job.setNumReduceTasks(1);
	    
	    // This section sets the values of the <K2, V2>
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(weka.classifiers.bayes.NaiveBayes.class);
	    job.setOutputValueClass(AggregateableEvaluation.class);
	    
	    // Set the input and output directories based on command line args
	    FileInputFormat.addInputPath(job, new Path(args[2]));
	    FileOutputFormat.setOutputPath(job, new Path(args[3]));
	    
	    // Set the input type of the environment
	    // (In this case we are overriding TextInputFormat)
	    job.setInputFormatClass(WekaInputFormat.class);
	    
	    // wait until the job is complete to exit
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
