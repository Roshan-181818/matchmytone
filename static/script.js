document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const resultCard = document.getElementById("result");

    fetch("/analyze", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        if(data.error){
            alert(data.error);
            return;
        }

        resultCard.classList.remove("hidden");

        document.getElementById("tone").innerText =
            "Detected Skin Tone: " + data.skin_tone;

        document.getElementById("match").innerText =
            "Compatibility: " + data.result;

        document.getElementById("scoreText").innerText =
            "Compatibility Score: " + data.score + "%";

        document.getElementById("progressBar").style.width =
            data.score + "%";

        const skinRGB = `rgb(${parseInt(data.skin_color[0])},
                             ${parseInt(data.skin_color[1])},
                             ${parseInt(data.skin_color[2])})`;

        const outfitRGB = `rgb(${parseInt(data.outfit_color[0])},
                               ${parseInt(data.outfit_color[1])},
                               ${parseInt(data.outfit_color[2])})`;

        document.getElementById("skinColorBox").style.background = skinRGB;
        document.getElementById("outfitColorBox").style.background = outfitRGB;
    });
});
