import './style.css';
import './app.css';

import {DoStuff} from '../wailsjs/go/main/App';

// Set up the greet function
window.doStuff = function () {
    DoStuff();
};
