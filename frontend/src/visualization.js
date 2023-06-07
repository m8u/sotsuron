import {mat4} from "gl-matrix";

let gl;

let models = [];
let currentModelIndex = -1;
let buffers;
let vertexCount = 0;
let height = 0;

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

function getLayerColor(layer) {
    switch (layer.Type) {
        case "Conv2D":
            return [].concat(...Array(6).fill([0.65625, 0.2734375, 0.625, 1.0]));
        case "MaxPooling2D":
            return [].concat(...Array(6).fill([0.13671875, 0.8046875, 0.41796875, 1.0]));
        case "FC":
            return [].concat(...Array(6).fill([0.15234375, 0.17578125, 0.17578125, 1.0]));
    }
}

function getLayerVertices(layer, lastY) {
    switch (layer.Type) {
        case "Conv2D":
        case "MaxPooling2D":
            return [
                layer.Width/2, lastY,
                -layer.Width/2, lastY,
                layer.Width/2, lastY-layer.Height,
                -layer.Width/2, lastY,
                layer.Width/2, lastY-layer.Height,
                -layer.Width/2, lastY-layer.Height,
            ];
        case "FC":
            return [
                layer.Output/2, lastY,
                -layer.Output/2, lastY,
                layer.Output/2, lastY-1,
                -layer.Output/2, lastY,
                layer.Output/2, lastY-1,
                -layer.Output/2, lastY-1,
            ];
    }
}

function initBuffers() {
    if (currentModelIndex === -1) {
        return;
    }
    const model = models[currentModelIndex];
    console.log(model);

    vertexCount = 0;
    const positions = [], colors = [];
    let lastY = 0;
    for (const layer of model) {
        positions.push(...getLayerVertices(layer, lastY));
        colors.push(...getLayerColor(layer));
        lastY -= (layer.Type === "FC" ? 1 : layer.Height) + 0.1;
        vertexCount += 6;
    }
    height = -lastY;

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
    gl.clearColor(0.97, 0.97, 0.97, 1.0);
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

    const fieldOfView = (45 * Math.PI) / 180; // in radians
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = 1000.0;
    const projectionMatrix = mat4.create();

    // note: glmatrix.js always has the first argument
    // as the destination to receive the result.
    mat4.perspective(projectionMatrix, fieldOfView, aspect, zNear, zFar);

    // Set the drawing position to the "identity" point, which is
    // the center of the scene.
    const modelViewMatrix = mat4.create();

    // Now move the drawing position a bit to where we want to
    // start drawing the square.
    mat4.translate(
        modelViewMatrix, // destination matrix
        modelViewMatrix, // matrix to translate
        [0.0, height/2, -height*1.3]
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

    loop(programInfo, buffers);
}

export function pushBestLayers(bestLayers) {
    models.push(bestLayers)
    currentModelIndex++;
    initBuffers();
}