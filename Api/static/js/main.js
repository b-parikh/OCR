var httpRequest;

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
    box1.innerText = 'processing...';
    box2.innerText = '';
    box3.innerText = '';
}

const alertContents = () => {
    if (httpRequest.readyState === XMLHttpRequest.DONE) {
        if (httpRequest.status === 200) {
            let response = JSON.parse(httpRequest.responseText);
            console.log(response, typeof(response));
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
    startTypeSetting();
}

function startTypeSetting() {
    var HUB = MathJax.Hub;
    HUB.Queue(["Typeset", HUB, "latex"]);
  }