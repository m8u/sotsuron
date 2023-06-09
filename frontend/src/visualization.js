import {mat4} from "gl-matrix";

let gl;

const fieldOfView = (45 * Math.PI) / 180; // in radians
const zNear = 0.1;
const zFar = 1000.0;

let models = [[{Type: "Conv2D", Width: 10, Height: 5}, {Type: "MaxPooling2D", Width: 7, Height: 5}, {Type: "FC", Output: 128}, {Type: "FC", Output: 10}]];
let currentModelIndex = 0; //  todo -1
let buffers;
let vertexCount = 0;
let height = 0;
let translation;
let projectionMatrix, modelViewMatrix;

const gap = 0.2;
let maxFCOutput = 0;
const fcHeight = 1;
const fcMaxWidth = 20;

let zoom = 1.0;
let mouseDown = false;
let lastMouseX = null, lastMouseY = null;
let spanX = 0, spanY = 0;

const vsSource = `
    attribute vec4 aVertexPosition;
    attribute vec4 aVertexColor;

    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;

    varying lowp vec4 vColor;

    void main(void) {
      gl_Position = uProjectionMatrix * uModelViewMatrix * aVertexPosition;
      vColor = aVertexColor;
    }
  `;

const fsSource = `
    varying lowp vec4 vColor;

    void main(void) {
      gl_FragColor = vColor;
    }
  `;

function initShaderProgram(vsSource, fsSource) {
    const vertexShader = loadShader(gl.VERTEX_SHADER, vsSource);
    const fragmentShader = loadShader(gl.FRAGMENT_SHADER, fsSource);

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert(
            `Unable to initialize the shader program: ${gl.getProgramInfoLog(
                shaderProgram
            )}`
        );
        return null;
    }

    return shaderProgram;
}

function loadShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(
            `An error occurred compiling the shaders: ${gl.getShaderInfoLog(shader)}`
        );
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}


function getLayerX0X1Y0Y1(layer, lastY) {
    switch (layer.Type) {
        case "Conv2D":
        case "MaxPooling2D":
            return [-layer.Width/2, layer.Width/2, lastY, lastY-layer.Height];
        case "FC":
            return [(-layer.Output/maxFCOutput)*(fcMaxWidth/2), (layer.Output/maxFCOutput)*(fcMaxWidth/2), lastY, lastY-fcHeight];
    }
}

function getLayerColor(layer) {
    const b = layer.hovered ? 1.5 : 1.0;
    switch (layer.Type) {
        case "Conv2D":
            return [].concat(...Array(6).fill([b*0.65625, b*0.2734375, b*0.625, 1.0]));
        case "MaxPooling2D":
            return [].concat(...Array(6).fill([b*0.13671875, b*0.8046875, b*0.41796875, 1.0]));
        case "FC":
            return [].concat(...Array(6).fill([b*0.4, b*0.4, b*0.4, 1.0]));
    }
}

function initBuffers() {
    if (currentModelIndex === -1) {
        return;
    }
    const model = models[currentModelIndex];

    // find max FC width and set it as width
    maxFCOutput = 0;
    for (let layer = model[model.length-1], i = model.length-1; i >= 0; i--, layer = model[i]) {
        if (layer.Type === "FC") {
            if (layer.Output > maxFCOutput) {
                maxFCOutput = layer.Output;
            }
        } else {
            break;
        }
    }

    vertexCount = 0;
    const positions = [], colors = [];
    let lastY = 0;
    let x0, x1, y0, y1
    for (let layer of model) {
        [x0, x1, y0, y1] = getLayerX0X1Y0Y1(layer, lastY);
        layer.x0 = x0;
        layer.x1 = x1;
        layer.y0 = y0;
        layer.y1 = y1;
        positions.push(x1, y0, x0, y0, x1, y1, x1, y1, x0, y0, x0, y1);
        colors.push(...getLayerColor(layer));
        lastY -= (layer.Type === "FC" ? fcHeight : layer.Height) + gap;
        vertexCount += 6;
    }
    height = -lastY - gap;

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    buffers = {
        position: positionBuffer,
        color: colorBuffer,
    };
}

function setPositionAttribute(programInfo) {
    const numComponents = 2; // pull out 2 values per iteration
    const type = gl.FLOAT; // the data in the buffer is 32bit floats
    const normalize = false; // don't normalize
    const stride = 0; // how many bytes to get from one set of values to the next
    // 0 = use type and numComponents above
    const offset = 0; // how many bytes inside the buffer to start from
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
    gl.vertexAttribPointer(
        programInfo.attribLocations.vertexPosition,
        numComponents,
        type,
        normalize,
        stride,
        offset
    );
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexPosition);
}

function setColorAttribute(programInfo) {
    const numComponents = 4;
    const type = gl.FLOAT;
    const normalize = false;
    const stride = 0;
    const offset = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.color);
    gl.vertexAttribPointer(
        programInfo.attribLocations.vertexColor,
        numComponents,
        type,
        normalize,
        stride,
        offset
    );
    gl.enableVertexAttribArray(programInfo.attribLocations.vertexColor);
}

function render(programInfo, buffers) {
    gl.clearColor(0.9, 0.9, 0.9, 1.0);
    gl.clearDepth(1.0); // Clear everything
    gl.enable(gl.DEPTH_TEST); // Enable depth testing
    gl.depthFunc(gl.LEQUAL); // Near things obscure far things

    // Clear the canvas before we start drawing on it.

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (currentModelIndex === -1) {
        return;
    }

    // Create a perspective matrix, a special matrix that is
    // used to simulate the distortion of perspective in a camera.
    // Our field of view is 45 degrees, with a width/height
    // ratio that matches the display size of the canvas
    // and we only want to see objects between 0.1 units
    // and 100 units away from the camera.
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    projectionMatrix = mat4.create();

    // note: glmatrix.js always has the first argument
    // as the destination to receive the result.
    mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);

    // Set the drawing position to the "identity" point, which is
    // the center of the scene.
    modelViewMatrix = mat4.create();

    // Now move the drawing position a bit to where we want to
    // start drawing the square.
    translation = [0.0 + spanX, height/2 + spanY, -(Math.max(fcMaxWidth, height/2/Math.tan(fieldOfView/2))) * zoom];
    mat4.translate(
        modelViewMatrix, // destination matrix
        modelViewMatrix, // matrix to translate
        translation
    ); // amount to translate

    setPositionAttribute(programInfo);
    setColorAttribute(programInfo);

    // Tell WebGL to use our program when drawing
    gl.useProgram(programInfo.program);

    // Set the shader uniforms
    gl.uniformMatrix4fv(
        programInfo.uniformLocations.projectionMatrix,
        false,
        projectionMatrix
    );
    gl.uniformMatrix4fv(
        programInfo.uniformLocations.modelViewMatrix,
        false,
        modelViewMatrix
    );

    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
}

function loop(programInfo) {
    render(programInfo, buffers);
    setTimeout(() => loop(programInfo, buffers), 1000 / 60);
}

export function initVisualization() {
    window.visualizationCanvas = document.querySelector("#visualization-canvas");
    window.dispatchEvent(new Event("resize"));

    gl = window.visualizationCanvas.getContext("webgl2");
    if (gl === null) {
        alert("Не удалось инициализировать WebGL. Возможно, ваш браузер не поддерживает его.");
        return;
    }

    const shaderProgram = initShaderProgram(vsSource, fsSource);
    const programInfo = {
        program: shaderProgram,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(shaderProgram, "aVertexPosition"),
            vertexColor: gl.getAttribLocation(shaderProgram, "aVertexColor"),
        },
        uniformLocations: {
            projectionMatrix: gl.getUniformLocation(shaderProgram, "uProjectionMatrix"),
            modelViewMatrix: gl.getUniformLocation(shaderProgram, "uModelViewMatrix"),
        },
    };

    initBuffers();

    window.visualizationCanvas.addEventListener("mousemove", (e) => {
        const rect = window.visualizationCanvas.getBoundingClientRect();
        let x = ((e.clientX - rect.left) / rect.width * 2 - 1) * -translation[2]/2 - spanX;
        let y = (e.clientY - rect.top) / rect.height * -1;

        if (height > fcMaxWidth) { // what the hell is this
            const heightCursed = height/2/Math.tan(fieldOfView/2);
            const min_new = -(height + (-translation[2] - heightCursed)/2);
            const max_new = -(0.0 - (-translation[2] - heightCursed)/2);
            y = ((max_new - min_new) * (y - -1.0) / (0.0 - (-1.0)) + min_new);
        } else {
            const aspect = rect.height / rect.width;
            const min_new = -(height + (-translation[2] * aspect - height)/2);
            const max_new = -(0.0 - (-translation[2] * aspect - height)/2);
            y = ((max_new - min_new) * (y - -1.0) / (0.0 - (-1.0)) + min_new);
        }
        y -= spanY;

        for (const layer of models[currentModelIndex]) {
            if (x >= layer.x0 && x <= layer.x1 && y <= layer.y0 && y >= layer.y1) {
                layer.hovered = true;
                initBuffers();
            } else if (layer.hovered) {
                layer.hovered = false;
                initBuffers();
            }
        }

        if (mouseDown) {
            if (lastMouseX == null) {
                lastMouseX = e.x;
                lastMouseY = e.y;
            }
            spanX += (e.x - lastMouseX) / window.visualizationCanvas.width * -translation[2]/2
            spanY -= (e.y - lastMouseY) / window.visualizationCanvas.height * -translation[2]/2;
            lastMouseX = e.x;
            lastMouseY = e.y;
        }
    });

    window.visualizationCanvas.addEventListener("mousedown", (e) => {
        mouseDown = true;
    });

    window.visualizationCanvas.addEventListener("mouseup", (e) => {
        mouseDown = false;
        lastMouseX = null;
        lastMouseY = null;
    });

    window.visualizationCanvas.addEventListener("mouseleave", (e) => {
        for (const layer of models[currentModelIndex]) {
            if (layer.hovered) {
                layer.hovered = false;
                initBuffers();
            }
        }
        mouseDown = false;
        lastMouseX = null;
        lastMouseY = null;
    });

    window.visualizationCanvas.addEventListener("wheel", (e) => {
        zoom += e.deltaY * 0.001;
    });

    loop(programInfo, buffers);
}

export function pushBestLayers(bestLayers) {
    models.push(bestLayers)
    currentModelIndex++;
    initBuffers();
}