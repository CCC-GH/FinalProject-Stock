// Select the button
var button = d3.select("#filter-btn");

// Select the form
var form = d3.select("#form");

// Create event handlers for clicking the button or pressing the enter key
button.on("click", runFilter);
form.on("submit",runFilter);


// ------------- PASS DATE INPUT & RUN TABLE FILTER FUNCTION ------------- //

// Create the function to run for both events
function runFilter() {

  // @app.route('/stockinteraction', methods=['GET', 'POST'])
  //   def access_stock():
  //   global stock
  //   global api
  //   if request.method == 'POST':
  //       stock = request.form['input']
  //       object = stock_info(stock)
  //   return render_template('index.html', embed=object)HTML:

    d3.csv("static/output/combinedDF.csv", function(data) {

        tabledata = data
        console.log(tabledata.length);
      
        // ------------- GET DATE FROM FORM INPUT ------------- //

        // Prevent the page from refreshing
        // d3.event.preventDefault();

        // Select the input element and get the raw HTML node
        var formInput = d3.select("#ticker-symbol");

        // Get the value property of the input element
        var inputTicker = formInput.property("value");

        console.log(inputTicker)

        // Get table element reference
        var table = document.getElementById("ticker-table-body");

        // Clear old table
        for (i = 0; i = table.rows.length; i++) {
          
          table.deleteRow(0);

        };
    
    // Populate table with filtered data
    
        for (var i = 0; i < tabledata.length; i++) {
        
          row = table.insertRow(i);

          // for (var j = 0; j < 7; j++) {

            cell_datetime = row.insertCell(0);
            cell_datetime.innerHTML = tabledata[i].Date;

            cell_city = row.insertCell(1);
            cell_city.innerHTML = Math.round(tabledata[i].Close * 100) / 100;

            cell_state = row.insertCell(2);
            cell_state.innerHTML = tabledata[i].Volume;

            cell_country = row.insertCell(3);
            cell_country.innerHTML = tabledata[i].Predict;

            cell_shape = row.insertCell(4);
            cell_shape.innerHTML = tabledata[i].SM1;

            cell_duration = row.insertCell(5);
            cell_duration.innerHTML = tabledata[i].SM2;

            cell_duration = row.insertCell(6);
            cell_duration.innerHTML = tabledata[i].SM3;

            cell_duration = row.insertCell(7);
            cell_duration.innerHTML = tabledata[i].SM4;

          // };

         };

        // Populate stock information panel
    
        selected = d3.select('#stock-info-fields')
        selected.html('')

        selected.append('p').text(`Company Name: ${infoDict.currInfo.symbol}`);
        selected.append('p').text(`Current Ask/Bid: ${infoDict.currInfo.ask} / ${infoDict.currInfo.bid}`);
        selected.append('p').text(`High/Low Price: ${infoDict.currInfo.dayHigh} / ${infoDict.currInfo.dayLow}`);
        selected.append('p').text(`Avg Volume: ${infoDict.currInfo.averageVolume}`);
        selected.append('p').text(`Volume: ${infoDict.currInfo.volume}`);
        selected.append('p').text(`52w High: ${infoDict.currInfo.fiftyTwoWeekHigh}`);
        selected.append('p').text(`52w Low: ${infoDict.currInfo.fiftyTwoWeekLow}`);
        selected.append('p').text(`MorningStar Rating: ${infoDict.currInfo.morningStarOverallRating}'])`);
        selected.append('p').text(`Short Ratio:: ${infoDict.currInfo.shortRatio}`);
    
     });


 };
