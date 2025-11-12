import React from 'react'
import styles from "./NavSection.module.css"
export default function NavSection() {
  return (
    <>
    <div className={styles.main}>
        <div className={styles.navbox}>

        <h1>
            Founder
        </h1>
        </div>
        <div className={styles.navbox}>

        <h1>
            Member
        </h1>
        </div>
    </div>
    </>
  )
}
