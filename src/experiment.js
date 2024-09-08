import { initJsPsych } from 'jspsych';
import jsPsychHtmlButtonResponse from '@jspsych/plugin-html-button-response';
import jsPsychHtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';

// Add the unhandled rejection handler here
window.addEventListener('unhandledrejection', function(event) {
  console.error('Unhandled promise rejection:', event.reason);
});

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables for countdown timer and required correct repetitions
let timeLimit = 30;  // Default time limit of 30 seconds for the entire experiment
let requiredCorrectRepetitions = 3;  // Default requirement to type the bigram correctly 3 times

let experimentStartTime;
let timerInterval;

// Load OSF API token
async function loadOSFToken() {
  try {
    const response = await fetch('./configs/token.json');
    const data = await response.json();
    console.log('OSF API token loaded');  // Log the loaded token
    return data.osf_api_token;
  } catch (error) {
    console.error('Error loading OSF token:', error);
    return null;
  }
}

// Load bigram pairs from a CSV file or text source
async function loadBigramPairs() {
  try {
    const response = await fetch('./bigram_pairs.csv');  // Replace with the correct path to your file
    const csvText = await response.text();
    const rows = csvText.split('\n').map(row => row.trim()).filter(row => row);
    return rows.map(row => {
      const bigrams = row.split(',').map(bigram => bigram.trim());
      return jsPsych.randomization.shuffle(bigrams);  // Randomly shuffle bigrams using the jsPsych instance
    });
  } catch (error) {
    console.error('Error loading bigram pairs:', error);
    return [];
  }
}

// Add global styles for the experiment
function setGlobalStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .jspsych-content {
      max-width: 90% !important;
      font-size: 24px !important;
    }
    .jspsych-btn {
      font-size: 20px !important;
      padding: 15px 25px !important;
      margin: 10px !important;
    }
    #timer {
      position: fixed;
      top: 10px;
      right: 20px;
      font-size: 20px;
      color: #000;
    }
  `;
  document.head.appendChild(style);
}

// Typing trial function
// Typing trial function
function createTypingTrial(bigram, bigramPair, trialId) {
  let keyData = [];
  let correctSequenceCount = 0;
  let typedSequence = "";
  const trialStartTime = performance.now();  // Start time of the trial

  function handleKeyPress(event) {
    const typedKey = event.key.toLowerCase();
    const expectedKey = bigram[typedSequence.length % bigram.length];

    const keydownTime = performance.now() - trialStartTime;  // Track keydown time

    // Log only the necessary fields: expectedKey, typedKey, and chosenBigram
    const keyLog = {
      trialId: trialId,
      bigramPair: bigramPair.join(", "),
      bigram: bigram,
      expectedKey: expectedKey,  // Save the expected key
      typedKey: typedKey,  // Save the typed key
      keydownTime: keydownTime.toFixed(2),  // Log the time of the key press
      chosenBigram: ""  // To be updated after the comfort choice trial
    };

    if (typedKey === expectedKey) {
      typedSequence += typedKey;
      document.querySelector('#user-input').textContent = typedSequence;  // Show typed sequence

      // Increase correct sequence count when a full bigram is typed
      if (typedSequence.length % bigram.length === 0) {
        correctSequenceCount++;
      }

      // If the required correct repetitions are reached, save the streak
      if (correctSequenceCount === requiredCorrectRepetitions) {
        keyData.push(keyLog);  // Log the final key event
        document.querySelector('#feedback').textContent = "Correct!";

        // End the trial after a short delay to give feedback
        setTimeout(() => {
          jsPsych.finishTrial({ keyData: keyData });  // Finish the trial and save key data
        }, 500);
      } else {
        // Log intermediate correct keys contributing to the streak
        keyData.push(keyLog);
      }
    } else {
      // If there's a mistake, reset everything and don't save any data
      typedSequence = "";
      correctSequenceCount = 0;
      document.querySelector('#user-input').textContent = "";  // Clear input
      document.querySelector('#error-message').textContent = "Mistake detected. Start over.";
      keyData = [];  // Clear key data since the sequence was broken
    }
  }

  return {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p>Type <b>${bigram}</b> ${requiredCorrectRepetitions} times in a row without mistakes.</p>
                   <p id="user-input" style="font-size: 24px; letter-spacing: 2px;"></p>
                   <p id="feedback" style="color: green;"></p>
                   <p id="error-message" style="color: red;"></p>
                 </div>
               </div>`,
    choices: "ALL_KEYS",
    response_ends_trial: false,
    data: {
      trialId: trialId,
      correctSequence: bigram,
      bigramPair: bigramPair.join(", "),
      keyData: []  // To store key events
    },
    on_load: function () {
      document.addEventListener('keydown', handleKeyPress);  // Attach keydown event
    },
    on_finish: function (data) {
      document.removeEventListener('keydown', handleKeyPress);  // Remove keydown event
      data.keyData = keyData;  // Attach key data to trial data
    }
  };
}

// Comfort choice trial
function createComfortChoiceTrial(bigram1, bigram2, trialIndex) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<p style="font-size: 28px;">Which pair was easier to type?</p>`,
    choices: [bigram1, bigram2, "No difference"],
    button_html: '<button class="jspsych-btn comfort-choice-button">%choice%</button>',
    data: {
      task: 'comfort_choice',
      bigram1: bigram1,
      bigram2: bigram2,
      trialId: `trial-${trialIndex}-choice`
    },
    on_finish: function (data) {
      let chosenBigram = "";

      if (data.response === 0) {
        chosenBigram = bigram1;
      } else if (data.response === 1) {
        chosenBigram = bigram2;
      } else {
        chosenBigram = "No difference";
      }

      // Update chosenBigram for all key logs in the successful streak
      jsPsych.data.get().filterCustom(trial => {
        return trial.trialId === `trial-${trialIndex}-1` || trial.trialId === `trial-${trialIndex}-2`;
      }).values().forEach(trial => {
        trial.keyData.forEach(key => {
          key.chosenBigram = chosenBigram;  // Set chosenBigram for keys in the streak
        });
      });
    }
  };
}

// Function to escape commas and wrap fields in quotes if necessary
// Function to escape commas and wrap fields in quotes if necessary
function escapeCSVField(field) {
  if (typeof field === 'string' && field.includes(',')) {
    return `"${field.replace(/"/g, '""')}"`;  // Escape double quotes by doubling them
  }
  return field;
}

// Function to convert data to CSV format
function convertToCSV(data) {
  const csvHeaders = ['trialId', 'bigramPair', 'bigram', 'expectedKey', 'typedKey', 'keydownTime', 'chosenBigram'];
  let csvContent = csvHeaders.join(',') + '\n';

  data.forEach(trial => {
    if (trial.keyData) {
      trial.keyData.forEach(keyEvent => {
        const row = [
          escapeCSVField(keyEvent.trialId || ''),
          escapeCSVField(keyEvent.bigramPair || ''),
          escapeCSVField(keyEvent.bigram || ''),
          escapeCSVField(keyEvent.expectedKey || ''),
          escapeCSVField(keyEvent.typedKey || ''),
          keyEvent.keydownTime !== undefined ? keyEvent.keydownTime : '',
          escapeCSVField(keyEvent.chosenBigram || '')
        ];
        csvContent += row.join(',') + '\n';
      });
    }
  });

  return csvContent;
}

// Function to store data on OSF
async function storeDataOnOSF(data, format = 'csv') {
  console.log("Received data for upload:", data);
  const osfToken = await loadOSFToken();
  console.log("Using OSF API token:", osfToken);

  if (!osfToken) {
    console.error('Error: OSF API token not available. Data will not be stored on OSF.');
    return;
  }

  const osfNodeId = "jf8sc";
  const fileExtension = format === 'csv' ? 'csv' : 'json';
  const createFileUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=experiment_data_${Date.now()}.${fileExtension}`;

  let fileData;
  let contentType;

  if (format === 'csv') {
    console.log("Converting data to CSV format...");
    fileData = convertToCSV(data);
    contentType = 'text/csv';
  } else {
    console.log("Converting data to JSON format...");
    fileData = JSON.stringify(data);
    contentType = 'application/json';
  }

  try {
    console.log(`Attempting to upload data to OSF. URL: ${createFileUrl}`);
    const uploadResponse = await fetch(createFileUrl, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${osfToken}`,
        'Content-Type': contentType,
      },
      body: fileData,
    });

    console.log("OSF Upload Response:", uploadResponse);

    if (!uploadResponse.ok) {
      const errorDetails = await uploadResponse.text();
      throw new Error(`Upload error! Status: ${uploadResponse.status}, Details: ${errorDetails}`);
    }

    console.log('Data successfully stored on OSF');
  } catch (error) {
    console.error('Error storing data on OSF:', error);
    throw error;  // Re-throw the error so it can be caught in endExperiment
  }
}

// End the experiment and upload data to OSF
function endExperiment() {
  console.log("endExperiment function is called...");
  const experimentData = jsPsych.data.get().values();
  console.log("Collected experiment data:", experimentData);

  // Log each trial's data for further confirmation
  experimentData.forEach((trial, index) => {
    console.log(`Trial ${index + 1} data:`, trial);
  });

  console.log("Attempting to store data on OSF...");
  storeDataOnOSF(experimentData, 'csv')
    .then(() => {
      console.log("Data storage process completed");
      // End the experiment after data is stored
      jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
    })
    .catch(error => {
      console.error("Error in storeDataOnOSF:", error);
      // End the experiment even if there's an error
      jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
    });
}

// Add end experiment screen
const thankYouTrial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p>Thank you for participating! Press the button to finish.</p>`,
  choices: ["Finish"],
  on_load: function() {
    console.log("Thank you trial loaded");
  },
  on_finish: function () {
    console.log("Thank you trial finished, calling endExperiment function now...");
    endExperiment();  // This will trigger the data storage process
    console.log("endExperiment function called, now ending the experiment");
    jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
  }
};

// Timer function for the entire experiment
function startExperimentTimer() {
  let timeRemaining = timeLimit;
  const timerElement = document.createElement('div');
  timerElement.id = 'timer';
  document.body.appendChild(timerElement);

  timerInterval = setInterval(() => {
    timeRemaining--;
    timerElement.textContent = `Time left: ${timeRemaining} s`;

    if (timeRemaining <= 0) {
      clearInterval(timerInterval);
      console.log("Time is up, ending experiment...");
      endExperiment();
      // Remove the jsPsych.endExperiment call from here
    }
  }, 1000);
}

// Start button screen
const startExperiment = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<p style="font-size: 28px;">Let's begin!</p>`,
  choices: ["Start"],
  button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
  on_finish: () => {
    experimentStartTime = performance.now();  // Set the start time
    startExperimentTimer();  // Start the timer for the entire experiment
  }
};

// Run the experiment
// Run the experiment
async function runExperiment() {
  setGlobalStyles();

  const osfToken = await loadOSFToken();
  const bigramPairs = await loadBigramPairs();
  if (bigramPairs.length === 0) {
    jsPsych.endExperiment('No bigram pairs available');
    return;
  }

  const randomizedBigramPairs = jsPsych.randomization.shuffle(bigramPairs);
  const timeline = [];

  // Add start screen to timeline
  timeline.push(startExperiment);

  // Create trials for each pair of bigrams
  randomizedBigramPairs.forEach(([bigram1, bigram2], index) => {
    timeline.push(createTypingTrial(bigram1, [bigram1, bigram2], `trial-${index + 1}-1`));
    timeline.push(createTypingTrial(bigram2, [bigram1, bigram2], `trial-${index + 1}-2`));
    timeline.push(createComfortChoiceTrial(bigram1, bigram2, index + 1));
  });

  timeline.push(thankYouTrial);

  // Run the timeline
  console.log("Running experiment timeline...");
  jsPsych.run(timeline);
}

// Start the experiment
runExperiment();