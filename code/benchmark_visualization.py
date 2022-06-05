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

<div class="alert alert-info" role="alert">
<b>Please note:</b> FLOP data needs to be manually set and is
therefore not generally available. Performance plots are only correct for the
(exact) KNN optimizations.
</div>

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
            #fig.set_figwidth(2*6.4)
            #fig.set_figheight(2*4.8)
            ax.set_facecolor('#F2F2F2')
            ax.grid(color='#FFFFFF', linestyle='-', linewidth=1.25)


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
                speedup_caption = " (max. speedup on n={}: {:.2f}x)".format(max_input_size, most_cycles/least_cycles)
            else:
                speedup_caption = ""

            ax.legend()
            ax.set_xlabel("n (input size)")
            ax.set_ylabel("cycles")
            pyplot.title("Runtime" + speedup_caption)

            tmp_path = tempfile.gettempdir() + "/asl_graph.png"
            pyplot.savefig(tmp_path, bbox_inches='tight')
            self.wfile.write(open(tmp_path, "rb").read())


        if self.path[:len("performance_plot")+2] == "/performance_plot/":
            ids_json = urllib.parse.unquote(self.path[len("performance_plot")+2:])
            ids = json.loads(ids_json)

            pyplot.clf()
            fig, ax = pyplot.subplots()
            #fig.set_figwidth(2*6.4)
            #fig.set_figheight(2*4.8)
            ax.set_facecolor('#F2F2F2')
            ax.grid(color='#FFFFFF', linestyle='-', linewidth=1.25)




            
            """
            # Measured flops for combined_knn_shapley_opt (incl. KNN)
            # fsize=2048
            flops_per_input_size = {
                256: 988787722.5,
                512: 2306199717.0,
                768: 4552783174.0,
                1024: 7558671523.0,
                1280: 11452999520.0,
                1536: 15886071111.5,
                1792: 21178901528.0,
                2048: 28407232030.5,
                2304: 35790876296.5,
                2560: 44297747210.0,
                2816: 53267698662.5,
                3072: 63361588574.5,
                3328: 74079016382.0,
                3584: 87117165072.0,
                3840: 97033511518.0,
                4096: 112495127061.5,
                4352: 126863856338.0,
                4608: 141662306489.0,
                4864: 158519563926.5,
                5120: 175211152093.0,
                5376: 192096612807.5,
                5632: 211809505841.5,
                5888: 231551384138.0,
                6144: 252365638445.5,
                6400: 273376111003.0,
                6656: 295685940731.5,
                6912: 319065620876.0,
                7168: 351751861224.5,
                7424: 367939920581.5,
                7680: 393364166482.5,
                7936: 420424704965.0,
                8192: 447644420964.0,
            }
            """


            
            # Measured flops for combined_shapley_opt but only KNN, no shapley
            # fsize = 2048
            flops_per_input_size = {
                256: 968388566,
                512: 2362149196,
                768: 4577232434,
                1024: 7552055390,
                1280: 11364591176,
                1536: 16097581073,
                1792: 21425685747,
                2048: 28097029330,
                2304: 35840395782,
                2560: 44085380478,
                2816: 53353882845,
                3072: 62934197937,
                3328: 74654523548,
                3584: 86878081347,
                3840: 96752826826,
                4096: 112118307320,
                4352: 126644757825,
                4608: 141527941701,
                4864: 158008989974,
                5120: 175112120131,
                5376: 193799109636,
                5632: 211823563799,
                5888: 230945947478,
                6144: 251985088690,
                6400: 273248801847,
                6656: 295238518804,
                6912: 319093974833,
                7168: 341092200761,
                7424: 367005191700,
                7680: 393023906163,
                7936: 419880370295,
                8192: 447407015180,
            }

        


            """
            # Measured flops for approx shapley without approx knn
            # fsize = 2048
            flops_per_input_size = {
                256: 889328980 -957208028,
                512: 2105432648 -2115595858,
                768: 4126305179 -4106834273,
                1024: 6943562675 - 6940876867,
                1280: 10568158248 - 10535426840,
                1536: 15056160666 - 14962891420,
                1792: 20248029335 - 20137737457,
                2048: 26277743238 - 26250671310,
                2304: 33126508565 - 33156554388,
                2560: 40681976527 - 40682636936,
                2816: 49172502109 - 49188679453,
                3072: 58523717428 - 58446374191,
                3328: 68600092760 - 68538681591,
                3584: 79452338546 - 79304545770,
                3840: 91157978490 - 91003537550,
                4096: 103636258933 - 103462736850,
                4352: 116935050368 - 116822161975,
                4608: 130989447773 - 130921262953,
                4864: 145853968621 - 145832086152,
                5120: 161569376528 - 161536939202,
                5376: 178045161317 - 177974477103,
                5632: 195492239204 - 195349165253,
                5888: 213497701888 - 213493779675,
                6144: 232523016134 - 232356631310,
                6400: 252257646726 - 252208994582,
                6656: 272737321341 - 272648722843,
                6912: 294895243881 - 292909989232,
                7168: 316201070336 - 316107763578,
                7424: 339289099027 - 339118005107,
                7680: 362998070902 - 362865693852,
                7936: 387511156247 - 387422011913,
                8192: 412767740741 - 412561114824,
            }
            """


            last_x = 0
            last_y = []
            for i in ids:
                runfile = runfiles[i]
                x = []
                y = []

                for input_size in runfile["input_sizes"]:
                    if int(input_size) < 1024:
                        continue
                    
                    flops = flops_per_input_size[input_size]
                    cycles = median(runfile["benchmarks"][str(input_size)])

                    x.append(input_size)
                    y.append(flops/cycles)
            
                last_x = runfile["input_sizes"][-1]
                last_y.append(y[-1])
                pyplot.plot(x, y, marker='^', label=runfile["label"])
            
            ax.legend()
            ax.set_xlabel("n (input size)")
            ax.set_ylabel("flops/cycle")
            #ax.set_xscale('log', base=2)

            speedup_caption = " (speedup at n={}: {:.2f}x)".format(last_x, max(last_y)/min(last_y))
            
            pyplot.title("Performance " + speedup_caption)
            #pyplot.suptitle("Performance " + speedup_caption)
            #pyplot.title(runfile["cpu"])

            tmp_path = tempfile.gettempdir() + "/asl_graph.png"
            pyplot.savefig(tmp_path, bbox_inches='tight')
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
