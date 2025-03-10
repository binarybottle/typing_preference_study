const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

/**
 * Generate a summary file from a raw data file
 * This script extracts information from raw data files without making assumptions
 * about user preferences
 * When the experiment was running, the user would have indicated their preference 
 * via the slider, which would determine the chosen/unchosen bigrams. 
 * Without that data, we can only extract the bigram pairs that were tested, not which one was preferred.
 * 
 * @param {string} rawFilePath - Path to the raw data CSV file
 * @param {string} outputFilePath - Path to save the summary file (optional, will use default naming if not provided)
 */
function generateSummaryFromRaw(rawFilePath, outputFilePath = null) {
  // Read the raw data file
  console.log(`Reading raw data file: ${rawFilePath}`);
  const rawData = fs.readFileSync(rawFilePath, 'utf8');
  
  // Parse the raw CSV data
  const parsedData = Papa.parse(rawData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
  });
  
  if (!parsedData.data || parsedData.data.length === 0) {
    console.error('Error: No data found in the raw file or invalid CSV format.');
    return;
  }
  
  console.log(`Successfully parsed ${parsedData.data.length} rows of raw data.`);
  
  // Group data by trialId
  const trialGroups = {};
  parsedData.data.forEach(row => {
    if (!row.trialId) return; // Skip rows without trialId
    
    if (!trialGroups[row.trialId]) {
      trialGroups[row.trialId] = {
        user_id: row.user_id,
        trialId: row.trialId,
        keyData: []
      };
    }
    
    trialGroups[row.trialId].keyData.push({
      expectedKey: row.expectedKey,
      typedKey: row.typedKey,
      isCorrect: row.isCorrect,
      keydownTime: row.keydownTime
    });
  });
  
  console.log(`Found ${Object.keys(trialGroups).length} unique trials.`);
  
  // Extract data for the summary
  const summaryData = [];
  
  Object.values(trialGroups).forEach(trial => {
    // Extract the text from the expected keys
    const text = trial.keyData.map(k => k.expectedKey).join('');
    
    // Analyze the text to find bigrams
    // Based on the experiment.js code, we know the structure contains:
    // random text + first bigram repeated + random text + second bigram repeated + random text + alternating bigrams
    
    // Count all bigrams in the text
    const bigramCounts = {};
    const bigramTimes = {};
    const bigramCorrect = {};
    
    for (let i = 0; i < trial.keyData.length - 1; i++) {
      // Form a bigram from consecutive expectedKeys
      const currentBigram = trial.keyData[i].expectedKey + trial.keyData[i+1].expectedKey;
      
      // Skip counting bigrams with spaces or non-letter characters
      if (currentBigram.length !== 2 || !/^[a-zA-Z]{2}$/.test(currentBigram)) {
        continue;
      }
      
      // Count occurrences of bigrams
      if (!bigramCounts[currentBigram]) {
        bigramCounts[currentBigram] = 0;
        bigramTimes[currentBigram] = [];
        bigramCorrect[currentBigram] = 0;
      }
      
      bigramCounts[currentBigram]++;
      
      // Calculate time between keypresses for correctly typed bigrams
      if (trial.keyData[i].isCorrect && trial.keyData[i+1].isCorrect) {
        const time = parseFloat(trial.keyData[i+1].keydownTime) - parseFloat(trial.keyData[i].keydownTime);
        if (!isNaN(time) && time > 0) {
          bigramTimes[currentBigram].push(time);
          bigramCorrect[currentBigram]++;
        }
      }
    }
    
    // Find the two most frequent bigrams
    // These should be the bigrams being tested in the experiment
    const sortedBigrams = Object.entries(bigramCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([bigram]) => bigram);
    
    // If we found at least two bigrams
    if (sortedBigrams.length >= 2) {
      const bigram1 = sortedBigrams[0];
      const bigram2 = sortedBigrams[1];
      
      // Calculate median times for each bigram
      const medianTime1 = calculateMedian(bigramTimes[bigram1] || []);
      const medianTime2 = calculateMedian(bigramTimes[bigram2] || []);
      
      // Create a row with empty fields for subjective data
      // We'll include both bigrams in the dataset without assuming which was chosen
      const summaryRow = {
        user_id: trial.user_id,
        trialId: trial.trialId,
        text: text,
        // Leave preference fields empty - don't make assumptions
        sliderValue: "",
        chosenBigram: "",
        unchosenBigram: "",
        chosenBigramTime: "",
        unchosenBigramTime: "",
        chosenBigramCorrect: "",
        unchosenBigramCorrect: "",
        // Include both identified bigrams and their metrics
        bigram1: bigram1,
        bigram2: bigram2,
        bigram1_time: medianTime1,
        bigram2_time: medianTime2,
        bigram1_correct: bigramCorrect[bigram1] || 0,
        bigram2_correct: bigramCorrect[bigram2] || 0
      };
      
      summaryData.push(summaryRow);
    } else {
      console.warn(`Warning: Could not find two distinct bigrams for trial ${trial.trialId}`);
    }
  });
  
  console.log(`Generated summary data for ${summaryData.length} trials.`);
  
  // Generate the output CSV
  const csvOutput = Papa.unparse(summaryData);
  
  // If output path is not provided, create one based on the input file
  if (!outputFilePath) {
    const rawFileName = path.basename(rawFilePath);
    const summaryFileName = rawFileName.replace('raw_data', 'summary_data');
    outputFilePath = path.join(path.dirname(rawFilePath), summaryFileName);
  }
  
  // Write to the output file
  fs.writeFileSync(outputFilePath, csvOutput);
  console.log(`Summary data saved to: ${outputFilePath}`);
  
  return {
    inputFile: rawFilePath,
    outputFile: outputFilePath,
    trialsProcessed: summaryData.length
  };
}

/**
 * Calculate the median of an array of numbers
 * @param {Array<number>} arr - Array of numbers
 * @returns {number|null} - Median value or null if the array is empty
 */
function calculateMedian(arr) {
  if (arr.length === 0) return null;
  const sorted = arr.sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// If script is run directly, process command line arguments
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node generate-summary.js <raw_data_file_path> [output_file_path]');
    process.exit(1);
  }
  
  const rawFilePath = args[0];
  const outputFilePath = args.length > 1 ? args[1] : null;
  
  try {
    generateSummaryFromRaw(rawFilePath, outputFilePath);
  } catch (error) {
    console.error('Error processing file:', error);
    process.exit(1);
  }
}

// Export function for use as a module
module.exports = { generateSummaryFromRaw };