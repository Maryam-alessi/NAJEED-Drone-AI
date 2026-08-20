// ============================================================
// NAJEED - AI EMERGENCY RESPONSE DASHBOARD
// DEMO / SIMULATION MODE
// ============================================================



// ============================================================
// PATIENT DATA
// ============================================================

const patients = [

    {
        id: "P-001",

        injuryType: "Visible Wound",

        injuryLocation: "Left Arm",

        bodyRegion: "Upper Limb",

        responseStatus: "Needs Attention",

        movement: "Detected",

        consciousness: "Responsive",

        priority: "Moderate",

        latitude: 24.7136,

        longitude: 46.6753
    },


    {
        id: "P-002",

        injuryType: "Visible Wound",

        injuryLocation: "Head",

        bodyRegion: "Head",

        responseStatus: "Immediate Assistance",

        movement: "Not Detected",

        consciousness: "Unresponsive",

        priority: "Critical",

        latitude: 24.7139,

        longitude: 46.6756
    },


    {
        id: "P-003",

        injuryType: "Visible Wound",

        injuryLocation: "Right Leg",

        bodyRegion: "Lower Limb",

        responseStatus: "Stable",

        movement: "Detected",

        consciousness: "Responsive",

        priority: "Low",

        latitude: 24.7133,

        longitude: 46.6751
    }

];



// ============================================================
// STATE
// ============================================================

let selectedPatientId = "P-001";

let droneConnected = false;

let missionRunning = false;



// ============================================================
// DOM ELEMENTS
// ============================================================

const patientCards =
    document.getElementById("patientCards");


const selectedPatientIdElement =
    document.getElementById("selectedPatientId");


const injuryTypeElement =
    document.getElementById("injuryType");


const injuryLocationElement =
    document.getElementById("injuryLocation");


const bodyRegionElement =
    document.getElementById("bodyRegion");


const responseStatusElement =
    document.getElementById("responseStatus");

const movementStatusElement =
    document.getElementById("movementStatus");


const consciousnessStatusElement =
    document.getElementById("consciousnessStatus");


const priorityBadge =
    document.getElementById("priorityBadge");


const latitudeValue =
    document.getElementById("latitudeValue");


const longitudeValue =
    document.getElementById("longitudeValue");


const missionState =
    document.getElementById("missionState");


const eventLog =
    document.getElementById("eventLog");


const connectBtn =
    document.getElementById("connectBtn");


const startMissionBtn =
    document.getElementById("startMissionBtn");


const landBtn =
    document.getElementById("landBtn");


const emergencyBtn =
    document.getElementById("emergencyBtn");


const clearLogBtn =
    document.getElementById("clearLogBtn");



// ============================================================
// HELPERS
// ============================================================

function getPatientById(patientId) {

    return patients.find(
        patient => patient.id === patientId
    );

}



function getPriorityClass(priority) {

    const value =
        priority.toLowerCase();


    if (value === "critical") {

        return "priority-critical";
    }


    if (value === "moderate") {

        return "priority-moderate";
    }


    return "priority-low";

}



function getPriorityBadgeClass(priority) {

    const value =
        priority.toLowerCase();


    if (value === "critical") {

        return "critical";
    }


    if (value === "moderate") {

        return "moderate";
    }


    return "low";

}



function formatLatitude(value) {

    const direction =
        value >= 0
            ? "N"
            : "S";


    return `${Math.abs(value).toFixed(4)}° ${direction}`;

}



function formatLongitude(value) {

    const direction =
        value >= 0
            ? "E"
            : "W";


    return `${Math.abs(value).toFixed(4)}° ${direction}`;

}



// ============================================================
// RENDER PATIENT CARDS
// ============================================================

function renderPatientCards() {

    patientCards.innerHTML = "";


    patients.forEach(patient => {

        const card =
            document.createElement("div");


        card.className =
            "patient-card";


        card.dataset.patientId =
            patient.id;


        if (
            patient.id ===
            selectedPatientId
        ) {

            card.classList.add(
                "selected"
            );

        }


        card.innerHTML = `

            <div class="patient-avatar">

                ${patient.id.split("-")[1]}

            </div>


            <div class="patient-card-info">

                <strong>
                    ${patient.id}
                </strong>

                <span>
                    ${patient.injuryLocation}
                </span>

                <span
                    class="
                        small-priority
                        ${getPriorityClass(patient.priority)}
                    "
                >

                    ${patient.priority.toUpperCase()}

                </span>

            </div>

        `;


        card.addEventListener(
            "click",
            () => {

                selectPatient(
                    patient.id
                );

            }
        );


        patientCards.appendChild(
            card
        );

    });

}



// ============================================================
// SELECT PATIENT
// ============================================================

function selectPatient(patientId) {

    const patient =
        getPatientById(patientId);


    if (!patient) {

        return;
    }


    selectedPatientId =
        patient.id;


    selectedPatientIdElement.textContent =
        patient.id;


    injuryTypeElement.textContent =
        patient.injuryType;


    injuryLocationElement.textContent =
        patient.injuryLocation;


    bodyRegionElement.textContent =
        patient.bodyRegion;


    responseStatusElement.textContent =
    patient.responseStatus;


    movementStatusElement.textContent =
        patient.movement;


    consciousnessStatusElement.textContent =
        patient.consciousness;


    priorityBadge.textContent =
        patient.priority.toUpperCase();


    priorityBadge.className =
        `priority-badge ${getPriorityBadgeClass(patient.priority)}`;


    latitudeValue.textContent =
        formatLatitude(
            patient.latitude
        );


    longitudeValue.textContent =
        formatLongitude(
            patient.longitude
        );


    renderPatientCards();


    updateCameraSelection();


    addLog(
    `${patient.id} selected — ${patient.injuryLocation}, ${patient.responseStatus}`
);

}



// ============================================================
// CAMERA BOX SELECTION
// ============================================================

function updateCameraSelection() {

    const boxes =
        document.querySelectorAll(
            ".detection-box"
        );


    boxes.forEach(box => {

        box.classList.remove(
            "selected"
        );


        if (
            box.dataset.patientId ===
            selectedPatientId
        ) {

            box.classList.add(
                "selected"
            );

        }

    });

}



document
    .querySelectorAll(".detection-box")
    .forEach(box => {

        box.addEventListener(
            "click",
            () => {

                selectPatient(
                    box.dataset.patientId
                );

            }
        );

    });



// ============================================================
// INCIDENT COUNTERS
// ============================================================

function updateIncidentOverview() {

    const totalPatients =
        patients.length;


    const criticalPatients =
        patients.filter(
            patient =>
                patient.priority === "Critical"
        ).length;


    const unresponsivePatients =
        patients.filter(
            patient =>
                patient.consciousness === "Unresponsive"
        ).length;


    document.getElementById(
        "patientCount"
    ).textContent =
        totalPatients;


    document.getElementById(
        "totalPatients"
    ).textContent =
        totalPatients;


    document.getElementById(
        "criticalPatients"
    ).textContent =
        criticalPatients;


    document.getElementById(
        "injuryCount"
    ).textContent =
        patients.length;


    document.getElementById(
        "unresponsivePatients"
    ).textContent =
        unresponsivePatients;

}



// ============================================================
// EVENT LOG
// ============================================================

function addLog(message) {

    const now =
        new Date();


    const time =
        now.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            }
        );


    const logItem =
        document.createElement("div");


    logItem.className =
        "log-item";


    logItem.innerHTML = `

        <span class="log-time">
            ${time}
        </span>

        <span class="log-dot"></span>

        <span class="log-message">
            ${message}
        </span>

    `;


    eventLog.prepend(
        logItem
    );

}



// ============================================================
// CONNECT DRONE
// DEMO SIMULATION
// ============================================================

connectBtn.addEventListener(
    "click",
    () => {

        droneConnected =
            !droneConnected;


        if (droneConnected) {

            connectBtn.textContent =
                "✓ DRONE CONNECTED";


            connectBtn.classList.add(
                "connected"
            );


            addLog(
                "Drone connection simulated successfully."
            );


            return;
        }


        connectBtn.textContent =
            "◉ CONNECT DRONE";


        connectBtn.classList.remove(
            "connected"
        );


        missionRunning =
            false;


        missionState.textContent =
            "STANDBY";


        addLog(
            "Drone disconnected."
        );

    }
);



// ============================================================
// START MISSION
// ============================================================

startMissionBtn.addEventListener(
    "click",
    () => {

        if (!droneConnected) {

            addLog(
                "Demo mission started without physical drone."
            );

        }
        else {

            addLog(
                "Mission started — drone scanning incident area."
            );

        }


        missionRunning =
            true;


        missionState.textContent =
            "SCANNING";


        setTimeout(
            () => {

                if (!missionRunning) {

                    return;
                }


                missionState.textContent =
                    "PATIENTS FOUND";


                addLog(
                    `${patients.length} patients detected in incident area.`
                );


                addLog(
                    "AI injury detection completed."
                );


                addLog(
                    "Simulated patient assessment generated."
                );


                addLog(
                    "Simulated GPS coordinates acquired."
                );

            },
            1200
        );

    }
);



// ============================================================
// LAND / RETURN
// ============================================================

landBtn.addEventListener(
    "click",
    () => {

        missionRunning =
            false;


        missionState.textContent =
            "RETURNING";


        addLog(
            "Return / landing command activated in demo mode."
        );


        setTimeout(
            () => {

                missionState.textContent =
                    "STANDBY";


                addLog(
                    "Mission returned to standby."
                );

            },
            1200
        );

    }
);



// ============================================================
// EMERGENCY STOP
// ============================================================

emergencyBtn.addEventListener(
    "click",
    () => {

        missionRunning =
            false;


        missionState.textContent =
            "EMERGENCY STOP";


        addLog(
            "EMERGENCY STOP activated."
        );

    }
);



// ============================================================
// CLEAR LOG
// ============================================================

clearLogBtn.addEventListener(
    "click",
    () => {

        eventLog.innerHTML =
            "";


        addLog(
            "Event log cleared."
        );

    }
);



// ============================================================
// INITIALIZE DASHBOARD
// ============================================================

function initializeDashboard() {

    renderPatientCards();


    updateIncidentOverview();


    selectPatient(
        selectedPatientId
    );


    addLog(
        "NAJEED emergency response dashboard initialized."
    );


    addLog(
        "AI patient detection module ready."
    );


    addLog(
        "Prototype mode active — waiting for mission."
    );

}



initializeDashboard();