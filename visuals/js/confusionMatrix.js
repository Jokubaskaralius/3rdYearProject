/* Confusion Matrix
Copyright (c) <Arpit Narechania>
<script src="https://gist.github.com/arpitnarechania/dbf03d8ef7fffa446379d59db6354bac.js"></script>
https://gist.github.com/arpitnarechania/dbf03d8ef7fffa446379d59db6354bac
MIT License */

var margin = { top: 50, right: 50, bottom: 100, left: 100 };

function Matrix(options) {
  var width = options.width,
    height = options.height,
    data = options.data,
    container = options.container,
    labelsData = options.labels,
    startColor = options.start_color,
    endColor = options.end_color,
    fold = options.fold;

  var widthLegend = 100;

  if (!data) {
    throw new Error("Please pass data");
  }

  if (!Array.isArray(data) || !data.length || !Array.isArray(data[0])) {
    throw new Error("It should be a 2-D array");
  }

  var maxValue = d3.max(data, function (layer) {
    return d3.max(layer, function (d) {
      return d;
    });
  });
  var minValue = d3.min(data, function (layer) {
    return d3.min(layer, function (d) {
      return d;
    });
  });

  var numrows = data.length;
  var numcols = data[0].length;

  var svg = d3
    .select(container)
    .style("width", "100%")
    .append("text")
    .text(`Fold number: ${fold}`)
    .attr("class", "text")
    .style("text-align", "center")
    .style("display", "block")
    .append("div")
    .attr("class", `entry-${fold}`)
    .style("display", "flex")
    .style("align-items", "center")
    .style("justify-content", "center")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var background = svg
    .append("rect")
    .style("stroke", "black")
    .style("stroke-width", "2px")
    .attr("width", width)
    .attr("height", height);

  var x = d3.scale.ordinal().domain(d3.range(numcols)).rangeBands([0, width]);

  var y = d3.scale.ordinal().domain(d3.range(numrows)).rangeBands([0, height]);

  var colorMap = d3.scale
    .linear()
    .domain([minValue, maxValue])
    .range([startColor, endColor]);

  var row = svg
    .selectAll(".row")
    .data(data)
    .enter()
    .append("g")
    .attr("class", "row")
    .attr("transform", function (d, i) {
      return "translate(0," + y(i) + ")";
    });

  var cell = row
    .selectAll(".cell")
    .data(function (d) {
      return d;
    })
    .enter()
    .append("g")
    .attr("class", "cell")
    .attr("transform", function (d, i) {
      return "translate(" + x(i) + ", 0)";
    });

  cell
    .append("rect")
    .attr("width", x.rangeBand())
    .attr("height", y.rangeBand())
    .style("stroke-width", 0);

  cell
    .append("text")
    .attr("dy", ".32em")
    .attr("x", x.rangeBand() / 2)
    .attr("y", y.rangeBand() / 2)
    .attr("text-anchor", "middle")
    .style("fill", function (d, i) {
      return d >= maxValue / 2 ? "white" : "black";
    })
    .text(function (d, i) {
      return d;
    });

  row
    .selectAll(".cell")
    .data(function (d, i) {
      return data[i];
    })
    .style("fill", colorMap);

  var labels = svg.append("g").attr("class", "labels");

  var colLabelHeight = height + 20;
  var columnLabels = labels
    .selectAll(".column-label")
    .data(labelsData)
    .enter()
    .append("g")
    .attr("class", "column-label")
    .attr("transform", function (d, i) {
      return "translate(" + x(i) + "," + colLabelHeight + ")";
    });

  columnLabels
    .append("line")
    .style("stroke", "black")
    .style("stroke-width", "1px")
    .attr("x1", x.rangeBand() / 2)
    .attr("x2", x.rangeBand() / 2)
    .attr("y1", 0)
    .attr("y2", 5);

  columnLabels
    .append("text")
    .attr("x", 30)
    .attr("y", y.rangeBand() / 2)
    .attr("dy", ".22em")
    .attr("text-anchor", "end")
    .attr("transform", "rotate(-60)")
    .text(function (d, i) {
      return d;
    });

  var rowLabels = labels
    .selectAll(".row-label")
    .data(labelsData)
    .enter()
    .append("g")
    .attr("class", "row-label")
    .attr("transform", function (d, i) {
      return "translate(" + 0 + "," + y(i) + ")";
    });

  rowLabels
    .append("line")
    .style("stroke", "black")
    .style("stroke-width", "1px")
    .attr("x1", 0)
    .attr("x2", -5)
    .attr("y1", y.rangeBand() / 2)
    .attr("y2", y.rangeBand() / 2);

  rowLabels
    .append("text")
    .attr("x", -8)
    .attr("y", y.rangeBand() / 2)
    .attr("dy", ".32em")
    .attr("text-anchor", "end")
    .text(function (d, i) {
      return d;
    });

  var key = d3
    .select(`.entry-${fold}`)
    .append("svg")
    .attr("width", widthLegend)
    .attr("height", height + margin.top + margin.bottom);

  var legend = key
    .append("defs")
    .append("svg:linearGradient")
    .attr("id", "gradient")
    .attr("x1", "100%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "100%")
    .attr("spreadMethod", "pad");

  legend
    .append("stop")
    .attr("offset", "0%")
    .attr("stop-color", endColor)
    .attr("stop-opacity", 1);

  legend
    .append("stop")
    .attr("offset", "100%")
    .attr("stop-color", startColor)
    .attr("stop-opacity", 1);

  key
    .append("rect")
    .attr("width", widthLegend / 2 - 10)
    .attr("height", height)
    .style("fill", "url(#gradient)")
    .attr("transform", "translate(0," + margin.top + ")");

  var y = d3.scale.linear().range([height, 0]).domain([minValue, maxValue]);

  var yAxis = d3.svg.axis().scale(y).orient("right");

  key
    .append("g")
    .attr("class", "y axis")
    .attr("transform", "translate(41," + margin.top + ")")
    .call(yAxis);
}

// The table generation function
function tabulate(data, columns) {
  if (data[0].Fold) var fold = data[0].Fold;
  var table = d3
      .select(`.entry-${fold}`)
      .append("table")
      .attr("style", "margin-left: " + margin.left + "px"),
    thead = table.append("thead"),
    tbody = table.append("tbody");

  // append the header row
  thead
    .append("tr")
    .selectAll("th")
    .data(columns)
    .enter()
    .append("th")
    .text(function (column) {
      return column;
    });

  // create a row for each object in the data
  var rows = tbody.selectAll("tr").data(data).enter().append("tr");

  // create a cell in each row for each column
  var cells = rows
    .selectAll("td")
    .data(function (row) {
      return columns.map(function (column) {
        return { column: column, value: row[column] };
      });
    })
    .enter()
    .append("td")
    .attr("style", "font-family: Courier") // sets the font style
    .html(function (d) {
      return d.value;
    });

  return table;
}

function tabulate_2(data, columns) {
  if (data[0].Fold) var fold = data[0].Fold;
  var table = d3
      .select("#container")
      .append("text")
      .text(`${data[0].title}`)
      .style("text-align", "center")
      .append("table")
      .attr("style", "margin: " + 5 + "px" + " " + margin.left + "px"),
    thead = table.append("thead"),
    tbody = table.append("tbody");

  // append the header row
  thead
    .append("tr")
    .selectAll("th")
    .data(columns)
    .enter()
    .append("th")
    .text(function (column) {
      return column;
    });

  // create a row for each object in the data
  var rows = tbody.selectAll("tr").data(data).enter().append("tr");

  // create a cell in each row for each column
  var cells = rows
    .selectAll("td")
    .data(function (row) {
      return columns.map(function (column) {
        return { column: column, value: row[column] };
      });
    })
    .enter()
    .append("td")
    .attr("style", "font-family: Courier") // sets the font style
    .html(function (d) {
      return d.value;
    });

  return table;
}
