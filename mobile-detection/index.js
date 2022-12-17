let video = document.querySelector("#videoElement");
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true})
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}

function start() {
  let canvas = document.getElementById("canvas2");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas
    .getContext("2d")
    .drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  /** Code to merge image **/
  /** For instance, if I want to merge a play image on center of existing image **/
  const playImage = new Image();
  playImage.addEventListener('load', () => console.log('loaded'))
  playImage.onload = () => {
    const startX = video.videoWidth / 2 - playImage.width / 2;
    const startY = video.videoHeight / 2 - playImage.height / 2;
    canvas
      .getContext("2d")
      .drawImage(playImage, startX, startY, playImage.width, playImage.height);
  };
  setTimeout(() => {
    const formData = new FormData();
    formData.append('file', convertFile(canvas.toDataURL('image/jpeg', 1.0)));


    const request = new XMLHttpRequest()
    request.onreadystatechange  = function(event) {
      console.log(event)
      if (request.readyState != XMLHttpRequest.DONE) {
        return
      }
      const {res} = JSON.parse(event.target.response)
      if(!res) return
      drawBox(res)
      window.myDetectProcess = setTimeout(start, 0)
    }
    request.open("POST", 'http://192.168.1.41:8000/')
    // request.setRequestHeader('Access-Control-Request-Headers', 'Content-Type')
    // request.setRequestHeader('Access-Control-Request-Method', 'DELETE')
    request.send(formData);
  }, 100)
  /** End **/
}

function convertFile(base64Str) {
  let types = base64Str.match(/image\/(jpeg|png|gif|bmp)/),
    arr = base64Str.split(','),
    bstr = atob(arr[1]),
    n = bstr.length,
    u8arr = new Uint8Array(n),
    now = Date.now(),
    blob = null
  if(!types) alert('error')
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }
  // return new File([u8arr], `image-copy-${Date.now()}.${types[1]}`, {type:types[0]})

  // fix IE
  blob = new Blob([u8arr], {type: types[0]})
  return new File([blob], `image-copy-${now}.${types[1]}`)
}

const boxCtn = document.getElementById('box-ctn')

let resetCount = 0
function drawBox(boxes) {
  if(boxes && boxes.length || resetCount > 7) {
    boxCtn.innerHTML = '';
  } else {
    resetCount++
  }
  boxes.forEach(rawBox => {
    const box = {
      left: rawBox[0],
      top: rawBox[1],
      width: rawBox[2] - rawBox[0],
      height: rawBox[3] - rawBox[1],
      scores: rawBox[4],
      classes: rawBox[5],
    }
    const $div = document.createElement('div')
    $div.className = 'rect'
    $div.style.top = box.top + 'px'
    $div.style.left = box.left + 'px'
    $div.style.width = box.width + 'px'
    $div.style.height = box.height + 'px'
    $div.innerHTML = `<span class='className'>${box.classes} ${box.scores}</span>`

    console.log(box)
    boxCtn.appendChild($div)
  })
}
setTimeout(start, 3000)
