
import "./Submit.css"

function Submit({btn1text, onClick}){
return(
    <input type="submit" onClick={onClick} value={btn1text} className="btn"/>
);
}

export {Submit}