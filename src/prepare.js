import { initJsPsych } from "jspsych";


export const settings = {
    colorOrder: Math.floor(Math.random()*2),
    pM: Array('easy', 'hard')[Math.floor(Math.random()*2)],
    pEM: [10, 10],
    gameTypeOrder: Math.floor(Math.random()*2),
    val: 3,
    nTrials: 50
}

export const text = {
    game1: settings.colorOrder == 1 ? 'Green Game' : 'Blue Game',
    color1: settings.colorOrder == 1 ? 'green' : 'blue',
    hex1: settings.colorOrder == 1 ? '#00aa00' : '#1067e8',
    span1: settings.colorOrder == 1 ? 'a-span' : 'b-span',
    game2: settings.colorOrder == 0 ? 'Green Game' : 'Blue Game',
    color2: settings.colorOrder == 0 ? 'green' : 'blue',
    hex2: settings.colorOrder == 0 ? '#00aa00' : '#1067e8',
    span2: settings.colorOrder == 0 ? 'a-span' : 'b-span',
    value: settings.val.toString(),
    plural: settings.val == 1 ? '' : 's', 
    wasWere: settings.val == 1 ? 'was' : 'were'
};

export default initJsPsych(); 