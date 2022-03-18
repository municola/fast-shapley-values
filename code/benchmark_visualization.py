import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from re import I
from threading import Thread
from time import sleep

# Read all runfiles
runfiles_paths = []
runfiles = []
runfiles_stripped = []

for f in os.listdir("./run/"):
    if f.endswith(".json"):
        runfiles_paths.append(f)

        rf = json.loads(open("./run/" + f, "r").read())
        
        # Todo: Determine median / best, etc. here!

        runfiles.append(rf)
        runfiles_stripped.append({
            "name" : f,
            "input_sizes" : rf["input_sizes"],
            "num_runs" : rf["num_runs"]
        })



header = """
<!DOCTYPE html>
<html>
    <head>
        <title>ASL - Benchmarks</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
    </head>
<body>
<nav class="navbar navbar-expand-lg navbar-light" style="background-color: #e3f2fd;">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">ASL Benchmarks</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item">
          <a class="nav-link active" aria-current="page" href="#">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Features</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Pricing</a>
        </li>
        <li class="nav-item">
          <a class="nav-link disabled">Disabled</a>
        </li>
      </ul>
    </div>
  </div>
</nav>
"""


main = """
<script>
  document.addEventListener('DOMContentLoaded', (e) => {
      var xhttp = new XMLHttpRequest();
      xhttp.onreadystatechange = function() {
          if (this.readyState == 4 && this.status == 200) {
              var runfiles = JSON.parse(xhttp.responseText);
              for(var i=0; i<runfiles["runfiles"].length; i++){
                  document.getElementById("runfiles").innerHTML += '<tr><th scope="row">' + i + '</th><td>' + runfiles["runfiles"][i]["name"] + '</td><td>' + runfiles["runfiles"][i]["input_sizes"] + '</td><td>' + runfiles["runfiles"][i]["num_runs"] + '</td></tr>';
              }
          }
      };
      xhttp.open("GET", "/runfiles", true);
      xhttp.send();
  });
</script>


<br><br>
<h5>Latest benchmarks</h5>
<br>

<div class="row">
<div class="col-6">
<table class="table table-striped">
  <thead>
    <tr>
      <th scope="col">#</th>
      <th scope="col">Name</th>
      <th scope="col">Input sizes</th>
      <th scope="col">Number of runs</th>
    </tr>
  </thead>
  <tbody id="runfiles">
  </tbody>
</table>
</div>

<div class="col-6">Plot will be here</div>
</div>
"""

footer = """
</body>
</html>
"""


class GETHandler(BaseHTTPRequestHandler):
    global runfiles

    def do_GET(self):
        self.send_response(200, "OK")
        self.end_headers()

        if self.path == "/":
            result = header + main + footer
        if self.path == "/runfiles":
            result = json.dumps(
              {
                "runfiles" : runfiles_stripped
              }
            )

        self.wfile.write(result.encode("utf-8"))


class Webinterface(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.httpd = HTTPServer(("0.0.0.0", 5002), GETHandler)
    
    def run(self):
        print("Webserver started on http://0.0.0.0:5002/")
        self.httpd.serve_forever(1)
    

# Start Webinterface in separate thread
webinterface = Webinterface()
webinterface.start()

# Main thread
while True:
    try:
        # Main logic goes here...


        sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        webinterface.httpd.server_close()
        webinterface.httpd.shutdown()
        webinterface.join()
        break
