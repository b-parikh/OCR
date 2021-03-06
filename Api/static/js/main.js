var httpRequest;
let resultImg = document.getElementById('resultImg');
let componentImg = document.getElementById('componentImg');
resultImg.style.visibility = 'hidden';
componentImg.style.visibility = 'hidden';

function analyze(file) {
    httpRequest = new XMLHttpRequest();

    if (!httpRequest) {
        alert('Giving up :( Cannot create an XMLHTTP instance');
        return false;
    }

    loading();

    httpRequest.onreadystatechange = alertContents;

    httpRequest.open('GET', `/api/${file}/analyze`, true);
    httpRequest.send();

}

function loading() {
    let box1 = document.getElementById('result');
    let box2 = document.getElementById('expr');
    let box3 = document.getElementById('latex');
    resultImg.style.visibility = 'hidden';
    componentImg.style.visibility = 'hidden';
    box1.innerText = 'processing...';
    box2.innerText = '';
    box3.innerText = '';
}

const alertContents = () => {
    if (httpRequest.readyState === XMLHttpRequest.DONE) {
        if (httpRequest.status === 200) {
            let response = JSON.parse(httpRequest.responseText);
            updatePage(response);
        } else {
            alert('There was a problem with the request.');
        }
    }
}

function updatePage(data) {
    let box1 = document.getElementById('result');
    let box2 = document.getElementById('expr');
    let box3 = document.getElementById('latex');

    box1.innerText = `results: [${data.results}]`;
    box2.innerText = `expr: ${data.expr}`;
    box3.innerText = `$\\text{Latex}$: ${data.expr}`;
    resultImg.setAttribute('src', `data:img/png;base64, ${data.result}`)
    componentImg.setAttribute('src', `data:img/png;base64, ${data.components}`)
    resultImg.style.visibility = 'visible';
    componentImg.style.visibility = 'visible';
    startTypeSetting();
}

function startTypeSetting() {
    var HUB = MathJax.Hub;
    HUB.Queue(["Typeset", HUB, "latex"]);
  }