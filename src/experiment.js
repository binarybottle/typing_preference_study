import { initJsPsych } from 'jspsych';
import htmlButtonResponse from '@jspsych/plugin-html-button-response';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import surveyMultiChoice from '@jspsych/plugin-survey-multi-choice';
import 'jspsych/css/jspsych.css';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables for countdown timer and required correct repetitions
let experimentConfig = {
  timeLimit: 10,  // Default time limit of 10 seconds for the entire experiment
  requiredCorrectRepetitions: 3,  // Default requirement to type the bigram correctly 3 times
  useTimer: false,  // Default to not using the timer
  practiceOnly: false  // If true, only run the practice set
};

let experimentStartTime;
let timerInterval;

// Prolific completion URL with placeholder for the completion code
const PROLIFIC_COMPLETION_URL = "https://app.prolific.co/submissions/complete?cc=";
// Completion code for successful completion
const COMPLETION_CODE = "C1CL3V94";
// Completion code for no consent
const NO_CONSENT_CODE = "C15846F6";

// Get Prolific ID from URL parameters
let prolificID = '';
function getUrlParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}
prolificID = getUrlParam('PROLIFIC_PID') || 'unknown';

// Function to redirect to Prolific
function redirectToProlific(code) {
  console.log(`Redirecting to Prolific with code: ${code}`);
  window.location.href = PROLIFIC_COMPLETION_URL + code;
}

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
  async function loadFile(filename) {
    try {
      const response = await fetch(`./${filename}`);
      const csvText = await response.text();
      return csvText.split('\n').map(row => row.trim()).filter(row => row)
        .map(row => row.split(',').map(bigram => bigram.trim()));
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      return [];
    }
  }

  const introductoryPairs = await loadFile('bigram_3pairs.csv');
  const mainPairs = await loadFile('bigram_80pairs.csv');

  return { introductoryPairs, mainPairs };
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
  type: htmlButtonResponse,
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
  type: htmlButtonResponse,
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
  type: htmlButtonResponse,
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
      // If consent is not given, redirect to Prolific with the no consent code
      redirectToProlific(NO_CONSENT_CODE);
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
      if (correctSequenceCount === experimentConfig.requiredCorrectRepetitions) {
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
    type: htmlKeyboardResponse,
    stimulus: `<div class="jspsych-content-wrapper">
                 <div class="jspsych-content">
                   <p style="white-space: nowrap;">Type ${experimentConfig.requiredCorrectRepetitions} times:</p>
                   <p style="white-space: nowrap;"><b>${bigram}</b></p>
                   <p id="user-input" style="font-size: 24px; letter-spacing: 2px;"></p>
                   <p id="feedback" style="color: green;"></p>
                   <p id="error-message" style="color: red;"></p>
                 </div>
               </div>`,
    choices: "ALL_KEYS",
    response_ends_trial: false,
    data: {
      task: 'typing',
      trialId: trialId,
      correctSequence: bigram,
      bigramPair: bigramPair.join(", "),
      keyData: []
    },
    on_load: function () {
      document.addEventListener('keydown', handleKeyPress);
    },
    on_finish: function (data) {
      document.removeEventListener('keydown', handleKeyPress);
      data.keyData = keyData;
    }
  };
}

// Comfort choice trial
function createComfortChoiceTrial(bigram1, bigram2, trialIndex) {
  return {
    type: htmlButtonResponse,
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

      // Update the previous two typing trials with the chosen bigram
      const allData = jsPsych.data.get().values();
      const currentTrialIndex = allData.length - 1;
      
      for (let i = currentTrialIndex - 1; i >= currentTrialIndex - 2; i--) {
        if (allData[i].task === 'typing' && allData[i].keyData) {
          allData[i].keyData.forEach(keyEvent => {
            keyEvent.chosenBigram = chosenBigram;
          });
        }
      }
    }
  };
}

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
    if (trial.task === 'typing' && trial.keyData) {
      trial.keyData.forEach(keyEvent => {
        const row = [
          escapeCSVField(trial.trialId || ''),
          escapeCSVField(trial.bigramPair || ''),
          escapeCSVField(trial.correctSequence || ''),
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
  const createFileUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=experiment_data_${prolificID}_${Date.now()}.${fileExtension}`;

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
  console.log("All experiment data:", experimentData);

  // Log each trial's data for further confirmation
  experimentData.forEach((trial, index) => {
    console.log(`Trial ${index + 1} data:`, trial);
  });

  console.log("Attempting to store data on OSF...");
  storeDataOnOSF(experimentData, 'csv')
    .then(() => {
      console.log("Data storage process completed");
      redirectToProlific(COMPLETION_CODE);
    })
    .catch(error => {
      console.error("Error in storeDataOnOSF:", error);
      redirectToProlific(COMPLETION_CODE);
    });
}

// Add end experiment screen
const thankYouTrial = {
  type: htmlButtonResponse,
  stimulus: `<p>Thank you for participating! <br>The experiment is now complete.</p>`,
  choices: ["Finish"],
  on_load: function() {
    console.log("Thank you trial loaded");
  },
  on_finish: function () {
    console.log("Thank you trial finished, calling endExperiment function now...");
    endExperiment();
  }
};

// Timer function for the entire experiment
function startExperimentTimer() {
  if (!experimentConfig.useTimer) {
    console.log("Timer is disabled.");
    return;  // Exit the function if the timer is not to be used
  }

  let timeRemaining = experimentConfig.timeLimit;
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
    }
  }, 1000);
}

// Start button screen
const startExperiment = {
  type: htmlButtonResponse,
  stimulus: function() {
    let stimulusText = `<p style="font-size: 28px;">Ready to start?</p>`;
    if (experimentConfig.useTimer) {
      stimulusText += `<p style="font-size: 24px;">You will have ${experimentConfig.timeLimit} seconds to complete the experiment.</p>`;
    }
    stimulusText += `<p style="font-size: 24px;">Press the button when you're ready to begin!</p>`;
    return stimulusText;
  },
  choices: ["Start"],
  button_html: '<button class="jspsych-btn" style="font-size: 24px; padding: 15px 30px;">%choice%</button>',
  on_finish: () => {
    experimentStartTime = performance.now();  // Set the start time
    if (experimentConfig.useTimer) {
      startExperimentTimer();  // Start the timer for the entire experiment (if enabled)
    }
  }
};

// Run the experiment
async function runExperiment(options = {}) {
  // Update experimentConfig with provided options
  Object.assign(experimentConfig, options);

  setGlobalStyles();

  const osfToken = await loadOSFToken();
  const { introductoryPairs, mainPairs } = await loadBigramPairs();
  
  if (introductoryPairs.length === 0 || (mainPairs.length === 0 && !experimentConfig.practiceOnly)) {
    jsPsych.endExperiment('Error loading bigram pairs');
    return;
  }

  const randomizedMainPairs = experimentConfig.practiceOnly ? [] : jsPsych.randomization.shuffle(mainPairs);
  const timeline = [];

  // Add consent trial to timeline
  timeline.push(consentTrial);

  // Create the experiment timeline
  const experimentTimeline = {
    timeline: [
      keyboardLayoutInfo,
      typingInstructionsInfo,
      startExperiment,
      // Introductory pairs (always in the same order)
      ...introductoryPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, [bigram1, bigram2], `intro-trial-${index + 1}-1`),
        createTypingTrial(bigram2, [bigram1, bigram2], `intro-trial-${index + 1}-2`),
        createComfortChoiceTrial(bigram1, bigram2, `intro-${index + 1}`)
      ]),
    ],
    conditional_function: function() {
      // Only run this timeline if consent was given (i.e., the first option was selected)
      return jsPsych.data.get().last(1).values()[0].response === 0;
    }
  };

  // If not practice only, add transition screen and main pairs
  if (!experimentConfig.practiceOnly) {
    experimentTimeline.timeline.push(
      {
        type: htmlButtonResponse,
        stimulus: `<p>Great job! You've completed the introductory pairs.<br>
                   Now we'll move on to the main part of the experiment.</p>`,
        choices: ['Continue'],
      },
      ...randomizedMainPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, [bigram1, bigram2], `main-trial-${index + 1}-1`),
        createTypingTrial(bigram2, [bigram1, bigram2], `main-trial-${index + 1}-2`),
        createComfortChoiceTrial(bigram1, bigram2, `main-${index + 1}`)
      ])
    );
  }

  // Always add thank you trial at the end
  experimentTimeline.timeline.push(thankYouTrial);

  timeline.push(experimentTimeline);

  // Run the timeline
  console.log("Running experiment timeline...");
  jsPsych.run(timeline);
}

// Start the experiment with options
runExperiment({ practiceOnly: experimentConfig.practiceOnly, useTimer: experimentConfig.useTimer, timeLimit: experimentConfig.timeLimit, requiredCorrectRepetitions: experimentConfig.requiredCorrectRepetitions});  // Example usage
