import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from matplotlib import pyplot
from statistics import median 
import tempfile
from threading import Thread, Lock
from time import sleep

# Read all runfiles
runfiles_lock = Lock()
runfiles_paths = []
runfiles = {}
runfiles_stripped = []
new_runfiles_stripped = []


def read_runfiles():
    runfiles_lock.acquire()
    for f in os.listdir("./run/"):
        if f.endswith(".json") and not f in runfiles:
            try:
                rf = json.loads(open("./run/" + f, "r").read())
            except:
                continue

            # Only append, if parsing succeeds

            runfiles_paths.append(f)                

            # Compute median cycles for each runfile
            median_cycles = []
            for input_size in rf["input_sizes"]:
                    median_cycles.append(median(rf["benchmarks"][str(input_size)]))

            rf["median_cycles"] = median_cycles

            # Set label = filename, if not manually named
            if not "label" in rf.keys():
                rf["label"] = f

            # Todo: Determine median / best, etc. here!
            stripped = {
                "name" : f,
                "label" : rf["label"],
                "input_sizes" : rf["input_sizes"],
                "median_cycles" : rf["median_cycles"],
                "num_runs" : rf["num_runs"],
                "implementation" : rf["implementation"],
                "turbo_boost_disabled" : rf["turbo_boost_disabled"],
                "implementation_correct" : rf["implementation_correct"]
            }


            runfiles[f] = rf
            runfiles_stripped.append(stripped)
            new_runfiles_stripped.append(stripped)
        
    runfiles_lock.release()



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
        <!--
            <li class="nav-item">
            <a class="nav-link" href="#">Features</a>
            </li>
            <li class="nav-item">
            <a class="nav-link" href="#">Pricing</a>
            </li>
            <li class="nav-item">
            <a class="nav-link disabled">Disabled</a>
            </li>
        --!>
      </ul>
    </div>
  </div>
</nav>
"""


main = """
<script>
    var to_plot = [];
    var plot_mode = 'runtime';

    function toggle_plot(name){
        if(to_plot.includes(name)){
            to_plot.splice(to_plot.indexOf(name), 1);
        } else {
            to_plot.push(name);
        }

        document.getElementById("plotwindow").innerHTML = '<img src="/' + plot_mode + '_plot/' + encodeURI(JSON.stringify(to_plot)) + '">';
    }

    function get_table_entry(i, runfile, new_entry) {
        var new_badge = '';
        if(new_entry){
            new_badge = '<span class="badge bg-info">new</span>&nbsp;&nbsp;';
        }
        var boost_badge = '';
        console.log(runfile["turbo_boost_disabled"]);
        if(runfile["turbo_boost_disabled"]){
            boost_badge = '<span class="badge bg-success">no turbo boost</span>&nbsp;&nbsp;';
        } else {
            boost_badge = '<span class="badge bg-danger">turbo boost</span>&nbsp;&nbsp;';
        }
        
        var correct_badge = '';
        if(runfile["implementation_correct"]){
            correct_badge = '<span class="badge bg-success">correct</span>&nbsp;&nbsp;';
        } else {
            correct_badge = '<span class="badge bg-danger">incorrect</span>&nbsp;&nbsp;';
        }

        var impl_badge = '<span class="badge bg-secondary">' + runfile["implementation"] + '</span>&nbsp;&nbsp;';
        
        return '<tr><th scope="row">' + i + '</th><td><input class="form-check-input" id="' + runfile["label"] + '" type="checkbox" value="" onclick="javascript:toggle_plot(\\\'' + runfile["name"] + '\\\');"></td><td>' + runfile["label"] + '&nbsp;&nbsp;' + impl_badge + correct_badge + boost_badge + new_badge + '</td><td>' + runfile["input_sizes"] + '</td><td>' + runfile["median_cycles"]  + '</td><td>' + runfile["num_runs"] + '</td></tr>';
    }

    function do_runtime_plots() {
        plot_mode = 'runtime';
        document.getElementById("plotwindow").innerHTML = '<img src="/' + plot_mode + '_plot/' + encodeURI(JSON.stringify(to_plot)) + '">';
    }
    
    function do_performance_plots() {
        plot_mode = 'performance';
        document.getElementById("plotwindow").innerHTML = '<img src="/' + plot_mode + '_plot/' + encodeURI(JSON.stringify(to_plot)) + '">';
    }

    document.addEventListener('DOMContentLoaded', (e) => {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                var runfiles = JSON.parse(xhttp.responseText);
                for(var i=0; i<runfiles["runfiles"].length; i++){
                    document.getElementById("runfiles").innerHTML += get_table_entry(i, runfiles["runfiles"][i], false);
                }
            }
        };
        xhttp.open("GET", "/runfiles", true);
        xhttp.send();
    });


    setInterval(function() {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                var runfiles = JSON.parse(xhttp.responseText);
                var new_top = "";
                for(var i=0; i<runfiles["updates"].length; i++){
                    new_top += get_table_entry(i, runfiles["updates"][i], true);
                }

                if(new_top != ""){
                    document.getElementById("runfiles").innerHTML = new_top + document.getElementById("runfiles").innerHTML;
                }

                for(var i=0; i<to_plot.length; i++){
                    document.getElementById(to_plot[i]).checked = true;
                }
            }
        };
        xhttp.open("GET", "/updates", true);
        xhttp.send();
    }, 10000);

</script>


<br><br>
<h5>Select plot mode</h5>
<span class="btn btn-secondary" onclick="javascript:do_runtime_plots()">Runtime plots</span>
<span class="btn btn-secondary" onclick="javascript:do_performance_plots()">Performance plots</span><br><br>
<h5>Latest benchmarks</h5>
<div class="row">
<div class="col-8">
<table class="table table-striped">
  <thead>
    <tr>
      <th scope="col">#</th>
      <th scope="col"><input class="form-check-input" type="checkbox" value=""></th>
      <th scope="col">Name</th>
      <th scope="col">Input sizes</th>
      <th scope="col">Median of cycles (runtime)</th>
      <th scope="col">Number of runs</th>
    </tr>
  </thead>
  <tbody id="runfiles">
  </tbody>
</table>
</div>

<div class="col-4" id="plotwindow"></div>
</div>
"""

footer = """
</body>
</html>
"""


class GETHandler(BaseHTTPRequestHandler):
    global runfiles
    global runfiles_stripped
    global new_runfiles_stripped
    global runfiles_lock

    def do_GET(self):
        global runfiles
        global runfiles_stripped
        global new_runfiles_stripped
        global runfiles_lock

        self.send_response(200, "OK")
        self.end_headers()

        if self.path == "/":
            result = header + main + footer
            self.wfile.write(result.encode("utf-8"))


        if self.path == "/runfiles":
            runfiles_lock.acquire()
            runfiles_stripped.sort(key=(lambda elem: int(elem["name"][:-5])), reverse=True)
            result = json.dumps(
              {
                "runfiles" : runfiles_stripped
              }
            )
            new_runfiles_stripped = []
            runfiles_lock.release()

            self.wfile.write(result.encode("utf-8"))


        if self.path == "/updates":
            runfiles_lock.acquire()
            new_runfiles_stripped.sort(key=(lambda elem: int(elem["name"][:-5])), reverse=True)
            new_runfiles_stripped.reverse()
            result = json.dumps(
              {
                "updates" : new_runfiles_stripped
              }
            )
            new_runfiles_stripped = []
            runfiles_lock.release()

            self.wfile.write(result.encode("utf-8"))



        if self.path[:len("runtime_plot")+2] == "/runtime_plot/":
            ids_json = urllib.parse.unquote(self.path[len("runtime_plot")+2:])
            ids = json.loads(ids_json)

            pyplot.clf()
            fig, ax = pyplot.subplots()


            max_input_size_equal = True
            if len(ids) > 0:
                max_input_size = max(map((lambda x: int(x)), runfiles[ids[0]]["input_sizes"]))
                #raise Exception("input sizes: " + str(runfiles[ids[0]]["input_sizes"]))
                max_input_size_no = runfiles[ids[0]]["input_sizes"].index(max_input_size)
                least_cycles = runfiles[ids[0]]["median_cycles"][max_input_size_no]
                most_cycles = runfiles[ids[0]]["median_cycles"][max_input_size_no]
            else:
                max_input_size = None
                max_input_size_equal = False
                
            
            for i in ids:
                runfile = runfiles[i]
                x = runfile["input_sizes"]
                y = runfile["median_cycles"]
            
                # Find the max speedup only if the max input size is equal everywhere
                # (compare the speedup there)
                if max_input_size != max(runfile["input_sizes"]):
                    max_input_size_equal = False
                else:
                    max_input_size_no = runfile["input_sizes"].index(max_input_size)
                    least = runfile["median_cycles"][max_input_size_no]
                    most = runfile["median_cycles"][max_input_size_no]

                    if least < least_cycles:
                        least_cycles = least
                    if most > most_cycles:
                        most_cycles = most

                pyplot.plot(x, y, marker='^', label=runfile["label"])
            
            if max_input_size_equal:
                speedup_caption = " (Max. speedup on n={}: {:.2f}x)".format(max_input_size, most_cycles/least_cycles)
            else:
                speedup_caption = ""

            ax.legend()
            ax.set_xlabel("n (input size)")
            ax.set_ylabel("cycles")
            pyplot.title("Runtime" + speedup_caption)

            tmp_path = tempfile.gettempdir() + "/asl_graph.png"
            pyplot.savefig(tmp_path)
            self.wfile.write(open(tmp_path, "rb").read())


        if self.path[:len("performance_plot")+2] == "/performance_plot/":
            ids_json = urllib.parse.unquote(self.path[len("performance_plot")+2:])
            ids = json.loads(ids_json)

            pyplot.clf()
            fig, ax = pyplot.subplots()
            for i in ids:
                runfile = runfiles[i]
                x = runfile["input_sizes"]
                y = []
                for input_size in x:
                    x_trg = 10
                    x_tst = 5
                    flops = x_tst * (2 + (x_trg - 2) * 7)
                    cycles = median(runfile["benchmarks"][str(input_size)])
                    y.append(flops/cycles)
            
                pyplot.plot(x, y, marker='^', label=runfile["label"])
            
            ax.legend()
            ax.set_xlabel("n (input size)")
            ax.set_ylabel("flops/cycle")
            pyplot.title("Performance")

            tmp_path = tempfile.gettempdir() + "/asl_graph.png"
            pyplot.savefig(tmp_path)
            self.wfile.write(open(tmp_path, "rb").read())
        


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

        read_runfiles()
        sleep(5)

    except KeyboardInterrupt:
        print("Exiting...")
        webinterface.httpd.server_close()
        webinterface.httpd.shutdown()
        webinterface.join()
        break
