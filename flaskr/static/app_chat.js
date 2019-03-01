//
//
/*
botui.message.add({
    human: true,
    content: 'Hello World from human!'
});
*/
  //
  botui.message.add({
    content: 'Welcome, from AI bot!'
  });
  //
  botui.message.bot({ // show first message
    delay: 200,
    content: 'hello'
  }).then(() => {
    console.log("#then-1");
  });
  
/*
function add_msg(){
    var params = new URLSearchParams();
    params.append('intext', '1');
    axios.post('./test3', params)
    .then(response =>{
        console.log(response.data );
    });
}
*/
  
