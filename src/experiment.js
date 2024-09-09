import { initJsPsych } from 'jspsych';
import HtmlButtonResponse from '@jspsych/plugin-html-button-response';
import HtmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import SurveyMultiChoice from '@jspsych/plugin-survey-multi-choice';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables for countdown timer and required correct repetitions
let timeLimit = 10;  // Default time limit of 30 seconds for the entire experiment
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

// First informational page
const keyboardLayoutInfo = {
  type: HtmlButtonResponse,
  stimulus: `
    <div class='instruction'> 
      <p>It is expected that your keyboard has the following character layout:</p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/qwerty-layout.jpg" width="500" style="max-width: 100%;">
      </div>
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// Second informational page
const typingInstructionsInfo = {
  type: HtmlButtonResponse,
  stimulus: `
    <div class='instruction'> 
      <p>You will be asked to <strong>touch type</strong> a pair of letters three times with your left hand.
         Touch type as you normally would, with left fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>A</span><span id=keystroke>S</span><span id=keystroke>D</span><span id=keystroke>F</span>:</span></p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/touchtype.jpg" width="500" style="max-width: 100%;">
      </div>
      <p>For instance, you will be asked to type 
        <span style="white-space: nowrap;"><span id=keystroke>a</span><span id=keystroke>b</span></span> three times.</p>
      <p>If you type this correctly three times in a row,
      then you will be asked to type a 2nd pair such as
        <span style="white-space: nowrap;"><span id=keystroke>c</span><span id=keystroke>d</span></span> three times.</p>
      <p>If you type this correctly as well,
        then you will be asked which letter pair 
        <span style="white-space: nowrap;"> &mdash; <span id=keystroke>a</span><span id=keystroke>b</span> or 
        <span id=keystroke>c</span><span id=keystroke>d</span> &mdash; </span> 
        is <b>easier</b> (more comfortable) for you to type.</p> 
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// Consent trial
const consentTrial = {
  type: HtmlButtonResponse,
  stimulus: `
    <div class='instruction' style='text-align: left; max-width: 800px; margin: 0 auto;'> 
      <h2 style='text-align: center;'>Welcome</h2>
      <dl>
          <dt>Purpose</dt>
          <dd>The purpose of this study is to determine how comfortable different pairs of keys 
          are to type on computer keyboards to inform the design of future keyboard layouts.</dd>

          <dt>Expectations</dt>
          <dd>It is expected that you will be <b>touch typing</b> on a <b>QWERTY desktop computer keyboard</b>.</dd>

          <dt>Procedures</dt>
          <dd>If you choose to participate, you will be repeatedly asked to type two new pairs 
          of letters and report which pair is easier (more comfortable) to type. </dd>

          <dt>Risks</dt>
          <dd>There are no anticipated risks or discomforts from this research that 
          you would not normally have when typing on your own keyboard.</dd>

          <dt>Benefits</dt>
          <dd>There are no anticipated benefits to you from this research.</dd>

          <dt>Compensation</dt>
          <dd>If you decide to participate, you will be compensated for your participation.</dd>

          <dt>Participation</dt>
          <dd>Taking part or not in this research study is your decision. 
          You can decide to participate and then change your mind at any point.</dd>

          <dt>Contact Information</dt>
          <dd>If you have any questions about the purpose, procedures, or any other issues 
          related to this research study you may contact the Principal Investigator, 
          Dr. Arno Klein, at arno.klein@childmind.org. </dd>
      </dl>
      <p style='text-align: center; font-weight: bold; margin-top: 20px;'>
        Do you consent to participate in this study? <br> You must be 18 years of age or older to participate.
      </p>
    </div>
  `,
  choices: ["I consent", "I do not consent"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>',
  on_finish: function(data) {
    if (data.response === 1) {  // "I do not consent" is selected
      // If consent is not given, redirect to Prolific with a special code
      window.location.href = "https://app.prolific.co/submissions/complete?cc=XXXXXXX";
    }
  }
};

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
      document.querySelector('#error-message').textContent = "";  // Clear error message

      // Increase correct sequence count when a full bigram is typed
      if (typedSequence.length % bigram.length === 0) {
        correctSequenceCount++;
      }

      // If the required correct repetitions are reached, save the streak
      if (correctSequenceCount === requiredCorrectRepetitions) {
        keyData.push(keyLog);  // Log the final key event

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
      document.querySelector('#error-message').textContent = "Mistake detected. Try again.";
      keyData = [];  // Clear key data since the sequence was broken
    }
  }

  return {
    type: HtmlKeyboardResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p style="white-space: nowrap;">Type ${requiredCorrectRepetitions} times:</p>
                   <p style="white-space: nowrap;"><b>${bigram}</b></p>
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
    type: HtmlButtonResponse,
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
      jsPsych.endExperiment("The experiment has ended. <br>Thank you for your participation!");
    })
    .catch(error => {
      console.error("Error in storeDataOnOSF:", error);
      // End the experiment even if there's an error
      jsPsych.endExperiment("The experiment has ended. Thank you for your participation!");
    });
}

// Add end experiment screen
const thankYouTrial = {
  type: HtmlButtonResponse,
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
  type: HtmlButtonResponse,
  stimulus: `<p style="font-size: 28px;">Ready to start? <br> If so, press the button!</p>`,
  choices: ["Start"],
  button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
  on_finish: () => {
    experimentStartTime = performance.now();  // Set the start time
    startExperimentTimer();  // Start the timer for the entire experiment
  }
};

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

  // Add consent trial to timeline
  timeline.push(consentTrial);

  // Create a conditional timeline for the rest of the experiment
  const experimentTimeline = {
    timeline: [
      keyboardLayoutInfo,
      typingInstructionsInfo,
      startExperiment,
      ...randomizedBigramPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, [bigram1, bigram2], `trial-${index + 1}-1`),
        createTypingTrial(bigram2, [bigram1, bigram2], `trial-${index + 1}-2`),
        createComfortChoiceTrial(bigram1, bigram2, index + 1)
      ]),
      thankYouTrial
    ],
    conditional_function: function() {
      // Only run this timeline if consent was given (i.e., the first option was selected)
      return jsPsych.data.get().last(1).values()[0].response === 0;
    }
  };

  timeline.push(experimentTimeline);

  // Run the timeline
  console.log("Running experiment timeline...");
  jsPsych.run(timeline);
}

// Start the experiment
runExperiment();