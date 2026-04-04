let index = 0;
let total = 0;

let current = resetState();

function resetState() {
    return {
        scene_type: null,
        room_type: null,
        furnishing: null,
        attributes: [],
        bad: false
    };
}

async function loadImage() {
    const res = await fetch(`/get_image/${index}`);
    const data = await res.json();

    if (data.error) return;

    document.getElementById("image").src = "/images/" + data.image_path;

    document.getElementById("image-info").innerText =
        data.image_path;

    if (data.annotation && Object.keys(data.annotation).length > 0) {
        current = {
            scene_type: data.annotation.scene_type || null,
            room_type: data.annotation.room_type || null,
            furnishing: data.annotation.furnishing || null,
            attributes: data.annotation.attributes || [],
            bad: data.annotation.bad || false
        };
    } else {
        current = resetState();
    }

    updateUI();
    updateProgress();
}

function setScene(val) {
    current.scene_type = val;
    updateUI();
}

function setRoom(val) {
    current.room_type = val;
    updateUI();
}

function setFurnishing(val) {
    current.furnishing = val;
    updateUI();
}

function toggleAttr(el) {
    if (el.checked) {
        current.attributes.push(el.value);
    } else {
        current.attributes = current.attributes.filter(a => a !== el.value);
    }
}

function markBad() {
    current.bad = !current.bad;
    updateUI();
}

function highlight(id, condition) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle("active", condition);
}

function updateUI() {
    highlight("scene-interior", current.scene_type === "interior");
    highlight("scene-exterior", current.scene_type === "exterior");

    highlight("room-living_room", current.room_type === "living_room");
    highlight("room-bedroom", current.room_type === "bedroom");
    highlight("room-kitchen", current.room_type === "kitchen");
    highlight("room-bathroom", current.room_type === "bathroom");
    highlight("room-balcony", current.room_type === "balcony");
    highlight("room-room", current.room_type === "room");

    highlight("furn-furnished", current.furnishing === "furnished");
    highlight("furn-unfurnished", current.furnishing === "unfurnished");

    document.querySelectorAll("input[type=checkbox]").forEach(cb => {
        cb.checked = current.attributes.includes(cb.value);
    });
}

async function save() {
    await fetch("/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            image_path: document.getElementById("image").src.split("/images/")[1],
            labels: current
        })
    });
}

async function next() {
    await save();

    do {
        index++;
        if (index >= total) return;
        await loadImage();
    } while (
        document.getElementById("skipAnnotated").checked &&
        current.scene_type !== null
    );
}

async function prev() {
    await save();
    index--;
    if (index < 0) index = 0;
    loadImage();
}

function updateProgress() {
    document.getElementById("progress").innerText =
        `${index + 1} / ${total}`;
}

document.addEventListener("keydown", (e) => {
    if (e.key === "i") setScene("interior");
    if (e.key === "e") setScene("exterior");

    if (e.key === "1") setRoom("living_room");
    if (e.key === "2") setRoom("bedroom");
    if (e.key === "3") setRoom("kitchen");
    if (e.key === "4") setRoom("bathroom");
    if (e.key === "5") setRoom("balcony");
    if (e.key === "6") setRoom("room");

    if (e.key === "f") setFurnishing("furnished");
    if (e.key === "u") setFurnishing("unfurnished");

    if (e.key === "b") toggleKey("bright");
    if (e.key === "m") toggleKey("modern");
    if (e.key === "r") toggleKey("renovated");

    if (e.key === "x") markBad();

    if (e.key === "ArrowRight") next();
    if (e.key === "ArrowLeft") prev();
});

function toggleKey(attr) {
    if (current.attributes.includes(attr)) {
        current.attributes = current.attributes.filter(a => a !== attr);
    } else {
        current.attributes.push(attr);
    }
}

window.onload = async () => {
    total = TOTAL_IMAGES;
    loadImage();
};