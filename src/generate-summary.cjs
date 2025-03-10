const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

/**
 * Generate a summary file from a raw data file
 * npm install papaparse
 * node generate-summary.js path/to/raw_data_unknown_1741293337149.csv
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
  
  // Extract text from trials
  const summaryData = [];
  
  Object.values(trialGroups).forEach(trial => {
    // Extract the text from the expected keys
    const text = trial.keyData.map(k => k.expectedKey).join('');
    
    // Find bigrams in the trial
    // Typically, bigrams are pairs of characters that appear in the text
    // We'll look for repeated pairs in the text to identify the bigrams
    const bigramCounts = {};
    const bigramTimes = {};
    const bigramCorrect = {};
    
    for (let i = 0; i < trial.keyData.length - 1; i++) {
      const currentBigram = trial.keyData[i].expectedKey + trial.keyData[i+1].expectedKey;
      
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
    
    // Find the two most common bigrams (these should be the ones being tested)
    const sortedBigrams = Object.entries(bigramCounts)
      .filter(([bigram]) => bigram.length === 2) // Ensure we're only looking at actual bigrams
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
      
      // We don't have slider values from the raw data, so we'll need to infer them
      // Let's use the median times to determine which bigram was likely easier to type
      // (Lower median time suggests easier typing)
      let sliderValue = 0;
      let chosenBigram = bigram1;
      let unchosenBigram = bigram2;
      
      if (medianTime1 !== null && medianTime2 !== null) {
        if (medianTime1 < medianTime2) {
          // Bigram1 was faster, so it was likely the chosen one
          sliderValue = -50; // Arbitrary negative value suggesting preference for bigram1
        } else if (medianTime2 < medianTime1) {
          // Bigram2 was faster, so it was likely the chosen one
          sliderValue = 50; // Arbitrary positive value suggesting preference for bigram2
          chosenBigram = bigram2;
          unchosenBigram = bigram1;
        }
      }
      
      // Add to summary data
      summaryData.push({
        user_id: trial.user_id,
        trialId: trial.trialId,
        text: text,
        sliderValue: sliderValue,
        chosenBigram: chosenBigram,
        unchosenBigram: unchosenBigram,
        chosenBigramTime: chosenBigram === bigram1 ? medianTime1 : medianTime2,
        unchosenBigramTime: unchosenBigram === bigram2 ? medianTime2 : medianTime1,
        chosenBigramCorrect: bigramCorrect[chosenBigram] || 0,
        unchosenBigramCorrect: bigramCorrect[unchosenBigram] || 0
      });
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