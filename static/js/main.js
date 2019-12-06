function getBotResponse() {
    var rawText = $("#textInput").val();
    var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document
        .getElementById("userInput")
        .scrollIntoView({ block: "start", behavior: "smooth" });

    $.get("/get", { msg: rawText }).done(function(data) {
        var botHtml = '<p class="botText"><span>' + data + "</span></p>";
        $("#chatbox").append(botHtml);
        document
            .getElementById("userInput")
            .scrollIntoView({ block: "start", behavior: "smooth" });
    });
}

$("#textInput").keypress(function(e) {
    if ($("#textInput").val() !== "") {
        if (e.which == 13) {
            getBotResponse();
        }
    }
});

$("#userInput").on('click', "#chatButton", function(e) {
    if ($("#textInput").val() !== "") {
        getBotResponse();
    }
})