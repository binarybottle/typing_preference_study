import { initJsPsych } from 'jspsych';
import htmlButtonResponse from '@jspsych/plugin-html-button-response';
import htmlKeyboardResponse from '@jspsych/plugin-html-keyboard-response';
import 'jspsych/css/jspsych.css';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Global variables and configuration
let experimentConfig = {
  requiredCorrectRepetitions: 3,
  timeLimit: 10,
  useTimer: false,
  practiceOnly: true,
  randomizePairOrder: true,
  randomizeBigramsWithinPairs: false,
  trainingBigramFile: 'bigram_tables/bigram_3pairs_LH.csv',
  mainBigramFile: 'bigram_tables/bigram_2x80pairs_LH.csv',
  character_list: 'abcdefghijklmnopqrstuvwxyz',  // 'abcdefghijklmnopqrstuvwxyz,./',
  ncharacters: 6,
};

let experimentStartTime;
let timerInterval;

// Prolific completion URL with placeholder for the completion code
const PROLIFIC_COMPLETION_URL = "https://app.prolific.co/submissions/complete?cc=";
const COMPLETION_CODE = "C1CL3V94";
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
    const response = await fetch('./token.json');
    const data = await response.json();
    console.log('OSF API token loaded');
    return data.osf_api_token;
  } catch (error) {
    console.error('Error loading OSF token:', error);
    return null;
  }
}

// Load bigram pairs from a CSV file or text source
async function loadBigramPairs(trainingFile, mainFile) {
  async function loadFile(filename) {
    try {
      const response = await fetch(`./${filename}`);
      const csvText = await response.text();
      return csvText.split('\n').map(row => row.trim()).filter(row => row)
        .map(row => {
          return row.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g)
                    .map(entry => entry.replace(/^"(.*)"$/, '$1').trim());
        });
    } catch (error) {
      console.error(`Error loading ${filename}:`, error);
      return [];
    }
  }

  const introductoryPairs = await loadFile(trainingFile);
  const mainPairs = await loadFile(mainFile);

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
    .slider-container {
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 20px 0;
    }
    .slider {
      width: 300px;
      margin: 0 10px;
    }
    .slider.inactive {
      opacity: 0.5;
    }
  `;
  document.head.appendChild(style);
}

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
      redirectToProlific(NO_CONSENT_CODE);
    }
  }
};

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
      <p>You will be asked to <strong>touch type</strong> text containing pairs of letters.
         Touch type as you normally would, with left fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>A</span><span id=keystroke>S</span><span id=keystroke>D</span><span id=keystroke>F</span></span>
         and right fingers above the home row letters 
         <span style="white-space: nowrap;"><span id=keystroke>J</span><span id=keystroke>K</span><span id=keystroke>L</span><span id=keystroke>;</span></span>
         </p>
      <div style="display: flex; justify-content: center; margin: 20px 0;">
        <img src="https://binarybottle.com/typing/bigram-typing-comfort-experiment/images/touchtype.jpg" width="500" style="max-width: 100%;">
      </div>
      <p>After typing the text, you will be asked which pair of letters was easier to type,
         and how much easier it was.</p>
    </div>
  `,
  choices: ["Next >"],
  button_html: '<button class="jspsych-btn" style="font-size: 16px; padding: 10px 20px; margin: 0 10px;">%choice%</button>'
};

// Function to generate random text with spaces
function generateRandomText(ncharacters, character_list) {
  let text = '';
  let nextSpace = 5 + Math.floor(Math.random() * 4); // 5-8 characters
  for (let i = 0; i < ncharacters; i++) {
    if (i === nextSpace) {
      text += ' ';
      nextSpace = i + 5 + Math.floor(Math.random() * 4);
    } else {
      text += character_list[Math.floor(Math.random() * character_list.length)];
    }
  }
  return text.replace(/\s+/g, ' ').trim(); // Ensure only single spaces and trim any leading/trailing spaces
}

// Function to update the color of individual letters as they're typed
function updateLetterColors(index, color, bold = false) {
  const letterSpans = document.querySelectorAll('.letter');
  if (letterSpans[index]) {
    letterSpans[index].style.color = color;
    letterSpans[index].style.fontWeight = bold ? 'bold' : 'normal';
  }
}

// Function to create the typing trial
function createTypingTrial(bigram1, bigram2, trialId) {
  const text1 = generateRandomText(experimentConfig.ncharacters, experimentConfig.character_list);
  const text2 = generateRandomText(experimentConfig.ncharacters, experimentConfig.character_list);
  const text3 = generateRandomText(experimentConfig.ncharacters, experimentConfig.character_list);
  const bigramRepetition1 = (bigram1 + ' ').repeat(experimentConfig.requiredCorrectRepetitions).trim();
  const bigramRepetition2 = (bigram2 + ' ').repeat(experimentConfig.requiredCorrectRepetitions).trim();
  const alternatingBigrams = ((bigram1 + ' ' + bigram2 + ' ').repeat(experimentConfig.requiredCorrectRepetitions)).trim();
  
  const fullText = `${text1} ${bigramRepetition1} ${text2} ${bigramRepetition2} ${text3} ${alternatingBigrams}`.replace(/\s+/g, ' ');
  
  let typedSequence = "";
  let keyData = [];
  const trialStartTime = performance.now();

  function handleKeyPress(event) {
    let typedKey = event.key.toLowerCase();
    const expectedKey = fullText[typedSequence.length];
    const keydownTime = performance.now() - trialStartTime;

    if (event.key === 'Shift' || event.key.length > 1) {
      return;
    }

    const isCorrect = typedKey === expectedKey;
    
    keyData.push({
      expectedKey: expectedKey,
      typedKey: typedKey,
      isCorrect: isCorrect,
      keydownTime: keydownTime.toFixed(2)
    });

    if (isCorrect) {
      typedSequence += typedKey;

      if (typedSequence === fullText) {
        showSlider(bigram1, bigram2, trialId, fullText, keyData);
      }
    }

    updateDisplay();
  }

  function updateDisplay() {
    document.getElementById('text-to-type').innerHTML = fullText.split('').map((char, index) => {
      const isBigram = (fullText.substr(index, 2) === bigram1 || fullText.substr(index, 2) === bigram2 ||
                        fullText.substr(index - 1, 2) === bigram1 || fullText.substr(index - 1, 2) === bigram2);
      const isTyped = index < typedSequence.length;
      const style = [];
      
      if (isBigram) {
        style.push('font-weight: bold');
      }
      
      if (isTyped) {
        if (isBigram) {
          style.push('color: #606060');  // Darker gray for typed bigrams
        } else {
          style.push('color: #A0A0A0');  // Lighter gray for other typed text
        }
      }
      
      return `<span class="letter" style="${style.join(';')}">${char}</span>`;
    }).join('');
  }

  return {
    type: htmlKeyboardResponse,
    stimulus: `
      <div class="jspsych-content-wrapper">
        <div class="jspsych-content">
          <p>Type the following text:</p>
          <p id="text-to-type" style="font-size: 24px; letter-spacing: 2px; white-space: pre-wrap;"></p>
        </div>
      </div>`,
    choices: "NO_KEYS",
    on_load: function() {
      document.addEventListener('keydown', handleKeyPress);
      updateDisplay();
    },
    on_finish: function(data) {
      document.removeEventListener('keydown', handleKeyPress);
    }
  };
}

function showSlider(bigram1, bigram2, trialId, fullText, keyData) {
  const sliderHtml = `
    <p>Which was easier to type?</p>
    <div class="slider-container">
      <span>${bigram1}</span>
      <input type="range" min="-100" max="100" value="0" class="slider" id="comfortSlider">
      <span>${bigram2}</span>
    </div>
    <button id="nextButton" style="display: none;">Next</button>
  `;

  document.querySelector('.jspsych-content').innerHTML += sliderHtml;

  const slider = document.getElementById('comfortSlider');
  const nextButton = document.getElementById('nextButton');

  function activateSlider() {
    if (!slider.classList.contains('active')) {
      slider.classList.add('active');
      nextButton.style.display = 'block';
    }
  }

  slider.addEventListener('mousedown', activateSlider);
  slider.addEventListener('touchstart', activateSlider);
  slider.addEventListener('input', activateSlider);

  nextButton.addEventListener('click', function() {
    let sliderValue = parseInt(slider.value);
    
    let chosenBigram, unchosenBigram;
    if (sliderValue === 0) {
      // Randomly choose if the slider is at 0
      if (Math.random() < 0.5) {
        chosenBigram = bigram1;
        unchosenBigram = bigram2;
        sliderValue = -1;  // Slight preference for the left bigram
      } else {
        chosenBigram = bigram2;
        unchosenBigram = bigram1;
        sliderValue = 1;  // Slight preference for the right bigram
      }
    } else {
      chosenBigram = sliderValue < 0 ? bigram1 : bigram2;
      unchosenBigram = sliderValue < 0 ? bigram2 : bigram1;
    }

    const bigramData = calculateBigramTimes(keyData, bigram1, bigram2);

    jsPsych.finishTrial({
      task: 'typing_and_choice',
      trialId: trialId,
      text: fullText,
      sliderValue: sliderValue,
      chosenBigram: chosenBigram,
      unchosenBigram: unchosenBigram,
      chosenBigramTime: bigramData[chosenBigram].medianTime,
      unchosenBigramTime: bigramData[unchosenBigram].medianTime,
      chosenBigramCorrect: bigramData[chosenBigram].correctCount,
      unchosenBigramCorrect: bigramData[unchosenBigram].correctCount,
      keyData: keyData
    });
  });
}

function calculateBigramTimes(keyData, bigram1, bigram2) {
  const bigramTimes = {
    [bigram1]: [],
    [bigram2]: []
  };
  const bigramCorrect = {
    [bigram1]: 0,
    [bigram2]: 0
  };

  for (let i = 0; i < keyData.length - 1; i++) {
    const currentBigram = keyData[i].typedKey + keyData[i+1].typedKey;
    if ((currentBigram === bigram1 || currentBigram === bigram2) && 
        keyData[i].isCorrect && keyData[i+1].isCorrect) {
      const time = keyData[i+1].keydownTime - keyData[i].keydownTime;
      bigramTimes[currentBigram].push(parseFloat(time));
      bigramCorrect[currentBigram]++;
    }
  }

  return {
    [bigram1]: {
      medianTime: calculateMedian(bigramTimes[bigram1]),
      correctCount: bigramCorrect[bigram1]
    },
    [bigram2]: {
      medianTime: calculateMedian(bigramTimes[bigram2]),
      correctCount: bigramCorrect[bigram2]
    }
  };
}

function calculateMedian(arr) {
  if (arr.length === 0) return null;
  const sorted = arr.sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
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
  const rawHeaders = ['user_id', 'trialId', 'expectedKey', 'typedKey', 'isCorrect', 'keydownTime'];
  const summaryHeaders = ['user_id', 'trialId', 'text', 'sliderValue', 'chosenBigram', 'unchosenBigram', 'chosenBigramTime', 'unchosenBigramTime', 'chosenBigramCorrect', 'unchosenBigramCorrect'];

  let rawContent = rawHeaders.join(',') + '\n';
  let summaryContent = summaryHeaders.join(',') + '\n';

  data.forEach(trial => {
    if (trial.task === 'typing_and_choice') {
      // Raw data
      trial.keyData.forEach(keyEvent => {
        const rawRow = [
          prolificID,
          trial.trialId,
          escapeCSVField(keyEvent.expectedKey),
          escapeCSVField(keyEvent.typedKey),
          keyEvent.isCorrect,
          keyEvent.keydownTime
        ];
        rawContent += rawRow.join(',') + '\n';
      });

      // Summary data
      const summaryRow = [
        prolificID,
        trial.trialId,
        escapeCSVField(trial.text),
        trial.sliderValue,
        escapeCSVField(trial.chosenBigram),
        escapeCSVField(trial.unchosenBigram),
        trial.chosenBigramTime,
        trial.unchosenBigramTime,
        trial.chosenBigramCorrect,
        trial.unchosenBigramCorrect
      ];
      summaryContent += summaryRow.join(',') + '\n';
    }
  });

  return { rawContent, summaryContent };
}

// Functionunction to store data on OSF
async function storeDataOnOSF(data) {
  const osfToken = await loadOSFToken();
  if (!osfToken) {
    console.error('Error: OSF API token not available. Data will not be stored on OSF.');
    return;
  }

  const osfNodeId = "jf8sc";
  const { rawContent, summaryContent } = convertToCSV(data);

  const rawDataUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=raw_data_${prolificID}_${Date.now()}.csv`;
  const summaryDataUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=summary_data_${prolificID}_${Date.now()}.csv`;

  try {
    await uploadToOSF(rawDataUrl, rawContent, osfToken);
    await uploadToOSF(summaryDataUrl, summaryContent, osfToken);
    console.log('Data successfully stored on OSF');
  } catch (error) {
    console.error('Error storing data on OSF:', error);
    throw error;
  }
}

async function uploadToOSF(url, data, token) {
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'text/csv',
    },
    body: data,
  });

  if (!response.ok) {
    const errorDetails = await response.text();
    throw new Error(`Upload error! Status: ${response.status}, Details: ${errorDetails}`);
  }
}

// Timer function for the entire experiment
function startExperimentTimer() {
  if (!experimentConfig.useTimer) {
    console.log("Timer is disabled.");
    return;
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
    experimentStartTime = performance.now();
    if (experimentConfig.useTimer) {
      startExperimentTimer();
    }
  }
};

// End the experiment and upload data to OSF
function endExperiment() {
  const experimentData = jsPsych.data.get().values();
  console.log("All experiment data:", experimentData);

  // Filter out any empty or invalid data
  const validData = experimentData.filter(trial => trial.task === 'typing_and_choice' && trial.keyData && trial.keyData.length > 0);
  console.log("Valid data for CSV:", validData);

  storeDataOnOSF(validData)
    .then(() => {
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

// Run the experiment
async function runExperiment(options = {}) {
  // Update experimentConfig with provided options
  Object.assign(experimentConfig, options);

  setGlobalStyles();

  const osfToken = await loadOSFToken();
  const { introductoryPairs, mainPairs } = await loadBigramPairs(
    experimentConfig.trainingBigramFile,
    experimentConfig.mainBigramFile
  );
  
  if (introductoryPairs.length === 0 || (mainPairs.length === 0 && !experimentConfig.practiceOnly)) {
    jsPsych.endExperiment('Error loading bigram pairs');
    return;
  }

  // Apply randomization based on configuration
  let processedMainPairs = mainPairs;
  if (experimentConfig.randomizePairOrder) {
    processedMainPairs = jsPsych.randomization.shuffle(processedMainPairs);
  }
  if (experimentConfig.randomizeBigramsWithinPairs) {
    processedMainPairs = processedMainPairs.map(pair => jsPsych.randomization.shuffle(pair));
  }

  const timeline = [];

  // Add consent trial to timeline
  timeline.push(consentTrial);

  // Create the experiment timeline
  const experimentTimeline = {
    timeline: [
      keyboardLayoutInfo,
      typingInstructionsInfo,
      startExperiment,
      ...introductoryPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, bigram2, `intro-trial-${index + 1}`)
      ]),
    ],
    conditional_function: function() {
      // Only run this timeline if consent was given (i.e., the first option was selected)
      return jsPsych.data.get().last(1).values()[0].response === 0;
    }
  };

  // Add transition screen and main pairs if not practiceOnly
  if (!experimentConfig.practiceOnly) {
    experimentTimeline.timeline.push(
      {
        type: htmlButtonResponse,
        stimulus: `<p>Great job! You've completed the practice session.<br>
                    Now we'll move on to the main part of the experiment.</p>`,
        choices: ['Continue'],
      },
      ...processedMainPairs.flatMap(([bigram1, bigram2], index) => [
        createTypingTrial(bigram1, bigram2, `main-trial-${index + 1}`)
      ])
    );
  }

  // Add thank you trial at the end
  experimentTimeline.timeline.push(thankYouTrial);

  timeline.push(experimentTimeline);

  // Run the timeline
  console.log("Running experiment timeline...");
  jsPsych.run(timeline);
}

// Start the experiment with options
runExperiment({
  practiceOnly: experimentConfig.practiceOnly,
  useTimer: experimentConfig.useTimer,
  timeLimit: experimentConfig.timeLimit,
  requiredCorrectRepetitions: experimentConfig.requiredCorrectRepetitions,
  randomizePairOrder: experimentConfig.randomizePairOrder,
  randomizeBigramsWithinPairs: experimentConfig.randomizeBigramsWithinPairs,
  trainingBigramFile: experimentConfig.trainingBigramFile,
  mainBigramFile: experimentConfig.mainBigramFile,
  character_list: experimentConfig.character_list,
  ncharacters: experimentConfig.ncharacters
});