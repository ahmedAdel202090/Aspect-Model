document.addEventListener("DOMContentLoaded", function () {

  var reviewSubmit = document.getElementById('reviewSubmit');
  reviewSubmit.onclick = function () {
    const review = document.querySelector('#review').value;
    const request = new XMLHttpRequest();
    request.open('POST', '/api/predict/classname');
    request.onload = () => {
      const data = JSON.parse(request.responseText);
      const li = document.createElement('li');
      li.innerHTML = review + " : have rate = " + data;
      li.className = 'rate'
      document.querySelector('#results').append(li);
    };

    const data = new FormData();
    data.append("review", review);
    request.send(data);
    return false;
  };


  var arApiT = document.getElementById('arApiT');
  arApiT.onclick = function () {
    reviews = ["I am going to get two pound",
      "I have bought samsung s20 i think it's overrated",
      "I like this phone but really it's software is piece of shit",
      "It's ok",
      "I don't recommend any one to buy this phone its sucks really",
      "good phone, I want to have one of it"];

    const arrRequest = new XMLHttpRequest();
    arrRequest.open('POST', '/api/batch_predict/classname');
    arrRequest.onload = () => {
      const data = JSON.parse(arrRequest.responseText);

      for (var i = 0; i < reviews.length; i += 1) {
        const li = document.createElement('li');
        li.innerHTML = reviews[i] + " : have rate = " + data[i];
        li.className = 'rate';
        document.querySelector('#results').append(li);
      }
    };

    const data = new FormData();
    for (var i = 0; i < reviews.length; i++) {
      data.append("reviews", reviews[i]);
    }
    arrRequest.send(data);
    return false;
  }

});
