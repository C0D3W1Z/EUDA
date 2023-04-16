var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

$(window).on('load', function(){
  // $messages.mCustomScrollbar();
  setTimeout(function() {
    fakeMessage();
  }, 100);
});


function setDate(){
  d = new Date()
  if (m != d.getMinutes()) {
    m = d.getMinutes();
    $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
  }
}

function insertMessage() {
  msg = $('.message-input').val();
  if ($.trim(msg) == '') {
    return false;
  }
  $('<div class="message message-personal">' + msg + '</div>').appendTo($('.messages')).addClass('new');
  setDate();
  $('.message-input').val(null);
  updateScrollbar();
  setTimeout(function() {
    fakeMessage();
  }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
  insertMessage();
});

$(window).on('keydown', function(e) {
  if (e.which == 13) {
    insertMessage();
    return false;
  }
})

function fakeMessage() {
  if ($('.message-input').val() != '') {
    console.log("hi")
    return false;
  }
  
  var text2 = document.getElementById("textfrompython").textContent;
  console.log(text2)

  var ftext = document.getElementById("textfrompython2").textContent;
  console.log(ftext)
  
  var Fake = [
    "Hello, what's on your mind?",
    text2,
  ]

  if (text2 != "none") {
    Fake[0] = text2; // Replace the first message in the array with the contents of text2
  }
  
  $('<div class="message loading new"><figure class="avatar"><img src="/static/assets/img/EUDA.png" /></figure><span></span></div>').appendTo($('.messages'));



  setTimeout(function() {
    $('.message.loading').remove();
    $('<div class="message new"><figure class="avatar"><img src="/static/assets/img/EUDA.png" /></figure>' + Fake[i] + '</div>').appendTo($('.messages')).addClass('new');
    setDate();
    i++;
  }, 1000 + (Math.random() * 20) * 100);

}

// select the messages container
var messagesContainer = $('.messages-content');

// scroll the messages container to the bottom
function scrollMessagesToBottom() {
    messagesContainer.scrollTop(messagesContainer.prop("scrollHeight"));
}

// call the function to scroll the messages container to the bottom when a new message is added
$('.message-submit').on('click', function() {
    scrollMessagesToBottom();
});
