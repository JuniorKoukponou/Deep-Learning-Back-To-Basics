var express = require("express");
var app = express();
var PythonShell = require("python-shell");

app.use(function (req, res, next) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers"
  );
  return next();
});

app.get("/", function (req, res) {
  res.setHeader("Content-Type", "text/plain");

  console.log("req.query: ", req.query);

  const { x, y, model } = req.query;

  console.log("params:", x, y, model);

  var options = {
    mode: 'text',
    pythonPath: '/usr/local/bin/python3',
    //pythonPath: 'path/to/python',
    //pythonOptions: ['-u'], // get print results in real-time
    //scriptPath: 'path/to/my/scripts',
    args: [x, y, model]
  };

  console.log("options:", options);

  PythonShell.run("predict.py", options, function (err, result) {
    if (err) {
      console.log("err", err);
      res.send(err);
    }
    else {
      console.log("result", result);
      res.send(result);
    }
  });
});

app.listen(8080);
